import Foundation
import TurboFieldfare

/// Protocol for an MCP registry that exposes tools and can execute tool calls.
public protocol MCPRegistryProtocol: Sendable {
    func functionDefinitions() async -> [GFTokenizer.FunctionDefinition]
    func execute(toolCall: ParsedToolCall) async throws -> String
    func disconnect() async
}
