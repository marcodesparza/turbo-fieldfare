import Foundation
import TurboFieldfare
import TurboFieldfareMCP

/// Decorates an `AppInferenceClient` with an internal MCP tool loop.
/// When the model emits a tool call, this client executes it through the
/// configured `MCPRegistry`, appends the result, and re-generates until a
/// final answer is produced.
public final class MCPAppInferenceClient: AppInferenceClient, @unchecked Sendable {
    public static let defaultMaxToolRounds = 10

    private let inner: any AppInferenceClient
    private let registry: any MCPRegistryProtocol
    private let maxToolRounds: Int

    public init(inner: any AppInferenceClient,
                config: MCPConfiguration,
                maxToolRounds: Int = defaultMaxToolRounds) async throws {
        self.inner = inner
        self.registry = try await MCPRegistry(config: config)
        self.maxToolRounds = maxToolRounds
    }

    public convenience init(inner: any AppInferenceClient,
                            configPath: String,
                            maxToolRounds: Int = defaultMaxToolRounds) async throws {
        let config = try MCPConfiguration.load(fromPath: configPath)
        try await self.init(inner: inner, config: config, maxToolRounds: maxToolRounds)
    }

    init(inner: any AppInferenceClient,
         registry: any MCPRegistryProtocol,
         maxToolRounds: Int = defaultMaxToolRounds) {
        self.inner = inner
        self.registry = registry
        self.maxToolRounds = maxToolRounds
    }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached { [weak self] in
                guard let self else { return }
                do {
                    let mcpTools = await self.registry.functionDefinitions()
                    let initialMessages = request.messages ?? [
                        GFTokenizer.Message(role: .user, content: request.prompt)
                    ]
                    var messages = initialMessages
                    var tools = request.tools ?? []
                    for tool in mcpTools {
                        if !tools.contains(where: { $0.name == tool.name }) {
                            tools.append(tool)
                        }
                    }

                    for round in 0..<self.maxToolRounds {
                        let roundRequest = AppGenerationRequest(
                            modelDirectory: request.modelDirectory,
                            prompt: request.prompt,
                            maxNewTokens: request.maxNewTokens,
                            maxContextTokens: request.maxContextTokens,
                            temperature: request.temperature,
                            topK: request.topK,
                            topP: request.topP,
                            repetitionPenalty: request.repetitionPenalty,
                            runtimeOptions: request.runtimeOptions,
                            tools: tools.isEmpty ? nil : tools,
                            messages: messages)

                        var toolCalls: [ParsedToolCall]?
                        for try await event in self.inner.generate(roundRequest) {
                            if case .toolCalls(let calls) = event {
                                toolCalls = calls
                                continue
                            }
                            continuation.yield(event)
                        }

                        guard let calls = toolCalls, !calls.isEmpty else {
                            continuation.finish()
                            return
                        }

                        // Execute each tool call and build result messages.
                        var resultMessages: [GFTokenizer.Message] = []
                        for call in calls {
                            let resultText: String
                            do {
                                resultText = try await self.registry.execute(toolCall: call)
                            } catch {
                                resultText = "Error: \(error)"
                            }
                            resultMessages.append(GFTokenizer.Message(
                                role: .tool,
                                content: resultText,
                                toolCallID: call.id,
                                name: call.name))
                        }

                        let assistantMessage = GFTokenizer.Message(
                            role: .assistant,
                            content: nil,
                            toolCalls: calls.map {
                                GFTokenizer.HistoricalToolCall(
                                    id: $0.id,
                                    name: $0.name,
                                    arguments: $0.arguments)
                            })

                        messages = messages + [assistantMessage] + resultMessages

                        // Prevent infinite loops if the model keeps calling tools.
                        if round == self.maxToolRounds - 1 {
                            continuation.finish()
                            return
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                self.inner.cancel()
            }
        }
    }

    public func toolCount() async -> Int {
        await registry.functionDefinitions().count
    }

    public func cancel() {
        inner.cancel()
    }

    deinit {
        Task { [registry] in
            await registry.disconnect()
        }
    }
}

extension MCPAppInferenceClient: AppModelLifecycleClient {
    public func ensureLoaded(modelDirectory: URL,
                             maxContextTokens: Int,
                             options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        guard let lifecycle = inner as? AppModelLifecycleClient else {
            throw AppInferenceError.modelLoadFailed("inner client has no load lifecycle")
        }
        try await lifecycle.ensureLoaded(modelDirectory: modelDirectory,
                                         maxContextTokens: maxContextTokens,
                                         options: options,
                                         forceLogitsHead: forceLogitsHead,
                                         onState: onState)
    }

    public func unload() async {
        await (inner as? AppModelLifecycleClient)?.unload()
    }

    public func shutdownForTermination() {
        (inner as? AppModelLifecycleClient)?.shutdownForTermination()
    }

    public func resetConversation(epoch: UUID) async throws {
        guard let lifecycle = inner as? AppModelLifecycleClient else {
            throw AppInferenceError.modelLoadFailed("inner client has no load lifecycle")
        }
        try await lifecycle.resetConversation(epoch: epoch)
    }

    public func restoreConversation(
        _ lineage: AppConversationLineage,
        epoch: UUID,
        options: AppRuntimeOptions,
        maxContextTokens: Int,
        onPrefillProgress: @escaping @Sendable (Int, Int) -> Void
    ) async throws -> Int {
        guard let lifecycle = inner as? AppModelLifecycleClient else {
            throw AppInferenceError.modelLoadFailed("inner client has no load lifecycle")
        }
        return try await lifecycle.restoreConversation(
            lineage, epoch: epoch, options: options,
            maxContextTokens: maxContextTokens,
            onPrefillProgress: onPrefillProgress)
    }
}