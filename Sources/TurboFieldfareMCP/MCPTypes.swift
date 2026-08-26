import Foundation
import TurboFieldfare

/// Minimal MCP 2024-11-05 protocol types for a TurboFieldfareServer MCP client.
/// Supports only the `tools` capability over stdio JSON-RPC.

enum MCPProtocol {
    static let version = "2024-11-05"
}

// MARK: - JSON-RPC envelope

struct MCPJSONRPCRequest<T: Encodable & Sendable>: Encodable, Sendable {
    let jsonrpc = "2.0"
    let id: Int
    let method: String
    let params: T
}

struct MCPJSONRPCNotification<T: Encodable & Sendable>: Encodable, Sendable {
    let jsonrpc = "2.0"
    let method: String
    let params: T
}

struct MCPJSONRPCResponse<T: Decodable & Sendable>: Decodable, Sendable {
    let jsonrpc: String
    let id: Int?
    let result: T?
    let error: MCPJSONRPCError?
}

struct MCPJSONRPCError: Codable, Equatable, Sendable {
    let code: Int
    let message: String
    let data: MCPJSONValue?
}

public enum MCPError: Error, CustomStringConvertible, Equatable {
    case transport(String)
    case protocolViolation(String)
    case rpc(code: Int, message: String)
    case unexpectedResponse

    public var description: String {
        switch self {
        case .transport(let message):
            return "MCP transport error: \(message)"
        case .protocolViolation(let message):
            return "MCP protocol error: \(message)"
        case .rpc(let code, let message):
            return "MCP RPC error \(code): \(message)"
        case .unexpectedResponse:
            return "MCP unexpected response"
        }
    }
}

// MARK: - Initialize

struct MCPInitializeParams: Encodable, Sendable {
    let protocolVersion: String
    let capabilities: MCPClientCapabilities
    let clientInfo: MCPImplementation

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocolVersion"
        case capabilities
        case clientInfo
    }
}

struct MCPClientCapabilities: Encodable, Sendable {
    // Intentionally minimal; we only consume tools today.
}

struct MCPImplementation: Codable, Sendable {
    let name: String
    let version: String
}

struct MCPInitializeResult: Decodable, Sendable {
    let protocolVersion: String
    let capabilities: MCPServerCapabilities
    let serverInfo: MCPImplementation
}

struct MCPServerCapabilities: Decodable, Sendable {
    let tools: MCPToolsCapability?
}

struct MCPToolsCapability: Decodable, Sendable {
    let listChanged: Bool?
}

struct MCPEmptyParams: Encodable, Sendable {}

// MARK: - Tools

struct MCPToolsListParams: Encodable, Sendable {
    let cursor: String?
}

struct MCPToolsListResult: Decodable, Sendable {
    let tools: [MCPTool]
    let nextCursor: String?
}

public struct MCPTool: Decodable, Sendable {
    public let name: String
    public let description: String?
    public let inputSchema: MCPJSONValue

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case inputSchema = "inputSchema"
    }
}

// MARK: - Tool call

struct MCPCallToolParams: Encodable, Sendable {
    let name: String
    let arguments: [String: MCPJSONValue]?

    enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }
}

public struct MCPCallToolResult: Decodable, Sendable {
    public let content: [MCPContent]
    public let isError: Bool?

    public var text: String {
        content.compactMap { item -> String? in
            switch item {
            case .text(let textContent):
                return textContent.text
            case .image(let imageContent):
                return "[Image: \(imageContent.mimeType)]"
            }
        }.joined(separator: "\n")
    }
}

public enum MCPContent: Decodable, Sendable {
    case text(MCPTextContent)
    case image(MCPImageContent)

    enum CodingKeys: String, CodingKey {
        case type
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            self = .text(try MCPTextContent(from: decoder))
        case "image":
            self = .image(try MCPImageContent(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "unknown MCP content type '\(type)'")
        }
    }
}

public struct MCPTextContent: Decodable, Sendable {
    public let text: String
}

public struct MCPImageContent: Decodable, Sendable {
    public let data: String
    public let mimeType: String

    enum CodingKeys: String, CodingKey {
        case data
        case mimeType = "mimeType"
    }
}

// MARK: - Loose JSON value for schemas and arguments

/// Mirrors the project's `JSONValue` but uses standard coding keys so it can be
/// embedded directly in MCP envelopes without custom adapters.
public indirect enum MCPJSONValue: Codable, Equatable, Sendable {
    case object([String: MCPJSONValue])
    case array([MCPJSONValue])
    case string(String)
    case integer(Int64)
    case number(Double)
    case bool(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([MCPJSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: MCPJSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    /// Convert to the project's `JSONValue` for use in tool templates.
    func toJSONValue() -> JSONValue {
        switch self {
        case .object(let value):
            return .object(value.mapValues { $0.toJSONValue() })
        case .array(let value):
            return .array(value.map { $0.toJSONValue() })
        case .string(let value):
            return .string(value)
        case .integer(let value):
            return .integer(value)
        case .number(let value):
            return .number(value)
        case .bool(let value):
            return .bool(value)
        case .null:
            return .null
        }
    }

    /// Convert from the project's `JSONValue` when building MCP arguments.
    static func from(_ value: JSONValue) -> MCPJSONValue {
        switch value {
        case .object(let value):
            return .object(value.mapValues { MCPJSONValue.from($0) })
        case .array(let value):
            return .array(value.map { MCPJSONValue.from($0) })
        case .string(let value):
            return .string(value)
        case .integer(let value):
            return .integer(value)
        case .unsignedInteger(let value):
            return .integer(Int64(value))
        case .decimal(let value):
            return .number(Double(truncating: NSDecimalNumber(decimal: value)))
        case .number(let value):
            return .number(value)
        case .bool(let value):
            return .bool(value)
        case .null:
            return .null
        }
    }
}