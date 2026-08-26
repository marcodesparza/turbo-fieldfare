import Foundation

/// Parsed contents of an MCP configuration file (e.g. `mcp.json`).
public struct MCPConfiguration: Codable, Equatable, Sendable {
    public let servers: [String: MCPStdioServerConfig]

    enum CodingKeys: String, CodingKey {
        case servers = "mcpServers"
    }

    public init(servers: [String: MCPStdioServerConfig]) {
        self.servers = servers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.servers = try container.decode([String: MCPStdioServerConfig].self,
                                             forKey: .servers)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(servers, forKey: .servers)
    }

    /// Load and decode an MCP configuration file.
    public static func load(from url: URL) throws -> MCPConfiguration {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MCPConfiguration.self, from: data)
    }

    /// Load and decode an MCP configuration file from a path string.
    public static func load(fromPath path: String) throws -> MCPConfiguration {
        try load(from: URL(fileURLWithPath: path))
    }
}