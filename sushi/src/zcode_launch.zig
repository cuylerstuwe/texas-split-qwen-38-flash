//! ZCode 3.14's versioned Personal Provider Config contract. No model-name
//! matching: every chat row receives the serving endpoint's advertised budget.
const std = @import("std");

fn quoted(a: std.mem.Allocator, value: anytype) ![]u8 {
    return std.json.Stringify.valueAlloc(a, value, .{});
}

fn effortsFor(entry: anytype) []const []const u8 {
    if (@hasField(@TypeOf(entry), "efforts")) {
        if (entry.efforts) |values| if (values.len > 0) return values;
    }
    return &.{ "none", "low", "medium", "high" };
}

fn defaultEffort(entries: anytype, model: []const u8) []const u8 {
    for (entries) |e| {
        if (!std.mem.eql(u8, e.id, model)) continue;
        const values = effortsFor(e);
        for ([_][]const u8{ "medium", "high", "xhigh", "low", "max", "on" }) |want| {
            for (values) |value| if (std.mem.eql(u8, want, value)) return value;
        }
        return values[0];
    }
    return "medium";
}

pub fn configJson(a: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: anytype) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    const endpoint = try std.fmt.allocPrint(a, "{s}/v1", .{base_url});
    defer a.free(endpoint);
    const url = try quoted(a, endpoint);
    defer a.free(url);
    const selected = try quoted(a, model);
    defer a.free(selected);
    const effort = try quoted(a, defaultEffort(entries, model));
    defer a.free(effort);
    try out.print(a,
        \\{{"schemaVersion":1,"config":{{
        \\"providerOrder":["sushi"],
        \\"defaultModelSelection":{{"providerId":"sushi","modelId":{s},"options":{{"reasoningLevel":{s}}}}},
        \\"providerConfigRules":{{"providerRules":[{{"providerId":"sushi","providerName":"Sushi","enabled":true,"config":{{
        \\"group":"standard-personal","access":{{"type":"api-key","apiKey":"sushi"}},
        \\"api":{{"type":"openai-chat-completions","baseUrl":{s}}},"personalModelIds":[
    , .{ selected, effort, url });
    for (entries, 0..) |e, i| {
        const id = try quoted(a, e.id);
        defer a.free(id);
        try out.print(a, "{s}{s}", .{ if (i == 0) "" else ",", id });
    }
    try out.appendSlice(a, "]}}]},\"modelConfigRules\":{\"manualProviderModelRules\":[],\"providerModelRules\":[");
    for (entries, 0..) |e, i| {
        const id = try quoted(a, e.id);
        defer a.free(id);
        const levels = try quoted(a, effortsFor(e));
        defer a.free(levels);
        try out.print(a,
            \\{s}{{"providerId":"sushi","modelId":{s},"config":{{"enabled":true,
            \\"properties":{{"contextWindow":{d},"requiresMfjsToolSchema":false,
            \\"inputFormat":{{"supportsText":true,"supportsImage":{s},"supportsVideo":false,"supportsAudio":false,"supportsPdf":false}},
            \\"outputFormat":{{"supportsText":true}},"supportsToolCall":true,"supportsJsonSchemaOutput":false,
            \\"supportsNativeWebSearch":false,"supportsMidConversationSystem":false}},
            \\"optionSpecs":{{"reasoningLevel":{{"values":{s},"map":"{{\"reasoning_effort\": reasoningLevel}}"}},
            \\"maxOutputTokens":{{"max":{d},"map":"{{\"max_tokens\": maxOutputTokens}}"}}}}}}}}
        , .{ if (i == 0) "" else ",", id, e.budget.context, if (e.vision) "true" else "false", levels, e.budget.output });
    }
    try out.appendSlice(a, "]}}}\n");
    return out.toOwnedSlice(a);
}
