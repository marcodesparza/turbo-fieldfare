import Foundation

/// Configuration for launching one stdio MCP server.
public struct MCPStdioServerConfig: Codable, Equatable, Sendable {
    public let command: String
    public let args: [String]
    public let env: [String: String]?
    public let cwd: String?

    public init(command: String,
                args: [String] = [],
                env: [String: String]? = nil,
                cwd: String? = nil) {
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd
    }
}

/// A single stdio MCP client that speaks JSON-RPC with one child process.
public actor MCPStdioClient {
    private let config: MCPStdioServerConfig
    private let process: Process
    private let stdin: FileHandle
    private let stdout: FileHandle
    private var nextRequestID: Int = 1
    private var inFlight: [Int: CheckedContinuation<Data, Error>] = [:]
    private var readerTask: Task<Void, Never>?
    private var isInitialized = false
    private var isShuttingDown = false

    public init(config: MCPStdioServerConfig) throws {
        self.config = config

        guard FileManager.default.isExecutableFile(atPath: config.command) else {
            throw MCPError.transport("command is not executable: \(config.command)")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: config.command)
        process.arguments = config.args
        if let cwd = config.cwd, !cwd.isEmpty {
            process.currentDirectoryURL = URL(fileURLWithPath: cwd)
        }

        // Inherit the parent environment and overlay configured values.
        var environment = ProcessInfo.processInfo.environment
        if let env = config.env {
            for (key, value) in env {
                environment[key] = value
            }
        }
        process.environment = environment

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()

        self.process = process
        self.stdin = stdinPipe.fileHandleForWriting
        self.stdout = stdoutPipe.fileHandleForReading

        // Drain stderr in the background so the child process never blocks on it.
        Task {
            let stderr = stderrPipe.fileHandleForReading
            while true {
                let data = stderr.availableData
                if data.isEmpty { break }
            }
        }
    }

    deinit {
        readerTask?.cancel()
        // Best-effort cleanup if disconnect() was not called. Process is not
        // Sendable, so we rely on NSObject's unchecked sendability.
        let process = process
        Task { @Sendable in
            if process.isRunning {
                process.terminate()
            }
        }
    }

    /// Perform the MCP initialize handshake.
    public func connect() async throws {
        guard !isInitialized else { return }

        readerTask = Task { [weak self] in
            await self?.readLoop()
        }

        let params = MCPInitializeParams(
            protocolVersion: MCPProtocol.version,
            capabilities: MCPClientCapabilities(),
            clientInfo: MCPImplementation(name: "TurboFieldfareServer", version: "1.0"))

        let request = MCPJSONRPCRequest(
            id: allocateRequestID(),
            method: "initialize",
            params: params)

        let resultData = try await performRequest(request)
        let result = try JSONDecoder().decode(MCPInitializeResult.self, from: resultData)
        guard result.protocolVersion == MCPProtocol.version else {
            throw MCPError.protocolViolation(
                "server protocol version \(result.protocolVersion) is not supported")
        }

        let notification = MCPJSONRPCNotification(
            method: "notifications/initialized",
            params: MCPEmptyParams())
        try write(notification)

        isInitialized = true
    }

    /// List tools exposed by the MCP server.
    public func listTools() async throws -> [MCPTool] {
        try await ensureInitialized()

        var allTools: [MCPTool] = []
        var cursor: String? = nil
        repeat {
            let params = MCPToolsListParams(cursor: cursor)
            let request = MCPJSONRPCRequest(
                id: allocateRequestID(),
                method: "tools/list",
                params: params)
            let resultData = try await performRequest(request)
            let result = try JSONDecoder().decode(MCPToolsListResult.self, from: resultData)
            allTools.append(contentsOf: result.tools)
            cursor = result.nextCursor
        } while cursor != nil

        return allTools
    }

    /// Call a tool on the MCP server and return its text representation.
    public func callTool(name: String,
                         arguments: [String: MCPJSONValue]?) async throws -> MCPCallToolResult {
        try await ensureInitialized()

        let params = MCPCallToolParams(name: name, arguments: arguments)
        let request = MCPJSONRPCRequest(
            id: allocateRequestID(),
            method: "tools/call",
            params: params)
        let resultData = try await performRequest(request)
        return try JSONDecoder().decode(MCPCallToolResult.self, from: resultData)
    }

    /// Close stdin and terminate the child process.
    public func disconnect() async {
        isShuttingDown = true
        readerTask?.cancel()

        // Fail any in-flight requests.
        for (_, continuation) in inFlight {
            continuation.resume(throwing: MCPError.transport("client disconnected"))
        }
        inFlight.removeAll()

        do {
            try stdin.close()
        } catch {}
        terminateProcess()
    }

    // MARK: - Private

    private func ensureInitialized() async throws {
        if !isInitialized {
            try await connect()
        }
    }

    private func allocateRequestID() -> Int {
        let id = nextRequestID
        nextRequestID += 1
        return id
    }

    private func performRequest<T: Encodable & Sendable>(_ request: MCPJSONRPCRequest<T>) async throws -> Data {
        try await withTimeout(seconds: 60) { [self] in
            try await withCheckedThrowingContinuation { continuation in
                Task {
                    await self.registerContinuation(id: request.id, continuation: continuation)
                    do {
                        try await self.write(request)
                    } catch {
                        await self.resumeContinuation(id: request.id, with: .failure(error))
                    }
                }
            }
        }
    }

    private func registerContinuation(id: Int, continuation: CheckedContinuation<Data, Error>) {
        inFlight[id] = continuation
    }

    private func resumeContinuation(id: Int, with result: Result<Data, Error>) {
        guard let continuation = inFlight.removeValue(forKey: id) else { return }
        continuation.resume(with: result)
    }

    private func write<T: Encodable & Sendable>(_ request: MCPJSONRPCRequest<T>) throws {
        let data = try JSONEncoder().encode(request)
        try stdin.write(contentsOf: data + Data("\n".utf8))
    }

    private func write<T: Encodable & Sendable>(_ notification: MCPJSONRPCNotification<T>) throws {
        let data = try JSONEncoder().encode(notification)
        try stdin.write(contentsOf: data + Data("\n".utf8))
    }

    private func readLoop() async {
        var buffer = Data()
        while !Task.isCancelled {
            // Block until data is available or EOF; `availableData` returns
            // empty when nothing is ready, which would be misread as EOF.
            let chunk = stdout.readData(ofLength: 4096)
            if chunk.isEmpty {
                // EOF: fail remaining requests and exit.
                await handleEOF()
                return
            }
            buffer.append(chunk)

            // Split on newlines.
            while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = buffer.prefix(upTo: newlineIndex)
                buffer.removeSubrange(...newlineIndex)
                if !line.isEmpty {
                    await handleLine(Data(line))
                }
            }
        }
        await handleEOF()
    }

    private func handleLine(_ line: Data) async {
        // Try to decode as a generic response to route by id.
        struct MinimalResponse: Decodable {
            let id: Int?
            let error: MCPJSONRPCError?
        }

        guard let minimal = try? JSONDecoder().decode(MinimalResponse.self, from: line),
              let id = minimal.id else {
            // Notifications and malformed lines are ignored for now.
            return
        }

        guard let continuation = inFlight.removeValue(forKey: id) else {
            // Response for an unknown id; ignore.
            return
        }

        if let error = minimal.error {
            continuation.resume(throwing: MCPError.rpc(code: error.code, message: error.message))
            return
        }

        continuation.resume(returning: line)
    }

    private func handleEOF() async {
        for (_, continuation) in inFlight {
            continuation.resume(throwing: MCPError.transport("MCP server closed stdout"))
        }
        inFlight.removeAll()
    }

    private func terminateProcess() {
        if process.isRunning {
            process.terminate()
        }
    }

    private func withTimeout<T: Sendable>(seconds: TimeInterval,
                                          operation: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw MCPError.transport("request timed out after \(seconds)s")
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }
}