import Foundation
import TurboFieldfare

/// Registry that owns all configured MCP clients and routes tool calls to them.
public actor MCPRegistry: MCPRegistryProtocol {
    public struct ToolEntry: Sendable {
        public let serverName: String
        public let toolName: String
        public let description: String
        public let inputSchema: MCPJSONValue
    }

    private struct ClientEntry: Sendable {
        let name: String
        let client: MCPStdioClient
    }

    private let clients: [ClientEntry]
    private var toolsByName: [String: ToolEntry] = [:]

    public init(config: MCPConfiguration) async throws {
        var entries: [ClientEntry] = []
        for (name, serverConfig) in config.servers {
            let client = try MCPStdioClient(config: serverConfig)
            try await client.connect()
            entries.append(ClientEntry(name: name, client: client))
        }
        self.clients = entries

        // Aggregate tools and detect name collisions across servers.
        var seen: [String: ToolEntry] = [:]
        for entry in entries {
            let tools = try await entry.client.listTools()
            for tool in tools {
                guard !tool.name.isEmpty else { continue }
                if let existing = seen[tool.name] {
                    throw MCPError.protocolViolation(
                        "tool '\(tool.name)' is defined by both '\(existing.serverName)' and '\(entry.name)'")
                }
                seen[tool.name] = ToolEntry(
                    serverName: entry.name,
                    toolName: tool.name,
                    description: tool.description ?? "",
                    inputSchema: tool.inputSchema)
            }
        }
        self.toolsByName = seen
    }

    /// Convert aggregated tools to the tokenizer's function-definition format.
    public func functionDefinitions() async -> [GFTokenizer.FunctionDefinition] {
        toolsByName.values.compactMap { entry -> GFTokenizer.FunctionDefinition? in
            do {
                let parameters = try GemmaToolSchema.adapted(
                    entry.inputSchema.toJSONValue(),
                    toolName: entry.toolName)
                return GFTokenizer.FunctionDefinition(
                    name: entry.toolName,
                    description: entry.description,
                    parameters: parameters)
            } catch {
                // Skip tools whose schemas cannot be represented by the Gemma template.
                return nil
            }
        }
    }

    /// Execute a parsed tool call against the appropriate MCP server.
    public func execute(toolCall: ParsedToolCall) async throws -> String {
        guard let entry = toolsByName[toolCall.name] else {
            throw MCPError.protocolViolation("unknown MCP tool '\(toolCall.name)'")
        }
        guard let clientEntry = clients.first(where: { $0.name == entry.serverName }) else {
            throw MCPError.protocolViolation("MCP server '\(entry.serverName)' not found")
        }

        let arguments: [String: MCPJSONValue]?
        switch toolCall.arguments {
        case .object(let value):
            arguments = value.mapValues { MCPJSONValue.from($0) }
        case .null:
            arguments = nil
        default:
            throw MCPError.protocolViolation(
                "tool arguments for '\(toolCall.name)' must be an object")
        }

        let result = try await clientEntry.client.callTool(
            name: entry.toolName,
            arguments: arguments)
        return result.text
    }

    /// True if at least one MCP server is configured and connected.
    public var isEmpty: Bool {
        clients.isEmpty
    }

    /// Disconnect all MCP clients.
    public func disconnect() async {
        for entry in clients {
            await entry.client.disconnect()
        }
    }
}