// gepard-gates — CLI parity-gate lane for the Gepard Swift port.
// P2: CodecOps + TextRepetition bit-exact vs the oracle goldens (pure-Swift, no MLX).
// P3: GepardBackbone (Qwen3.5 full-attn lift) prefill parity vs the torch-CPU goldens.
// Usage:  swift run gepard-gates --p2 [p2_goldens.json]
//         swift run gepard-gates --p3 <model.safetensors> [p3_goldens.safetensors]
// Default golden paths are Goldens/* relative to CWD (the package dir).

import Foundation
import Darwin
import MLX
import MLXNN
import GepardCore
import MLXGepardTTS
import MLXToolKit

// MARK: - Golden bundle schema

struct P2Goldens: Codable {
    struct Special: Codable { let sot, eot, sos: Int }
    struct TRConfig: Codable { let enabled: Bool; let target_text_tokens, apply_below, max_repeats: Int }
    struct TRCase: Codable { let name: String; let ids: [Int]; let R: Int; let cond_ids: [Int] }
    struct Unfold: Codable { let packed: [[Int]]; let expected_ref_codes_TC: [[Int]] }
    struct Dequant: Codable { let codes: [[Int]]; let expected_TC: [[Double]] }
    let special_tokens: Special
    let textrep_config: TRConfig
    let textrep_cases: [TRCase]
    let fsq_levels: [Int]
    let num_layers: Int
    let unfold: Unfold
    let dequant: Dequant
}

func fail(_ msg: String) -> Never { FileHandle.standardError.write(Data("FAIL: \(msg)\n".utf8)); exit(1) }

// MARK: - Args

let args = CommandLine.arguments

// P3/P4 dispatch (MLX). Positional args after the flag: <model.safetensors> [goldens].
if args.contains("--p3") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard let modelPath = pos.first else {
        fail("--p3 needs <model.safetensors> [p3_goldens.safetensors]")
    }
    let gp = pos.dropFirst().first ?? "Goldens/p3_goldens.safetensors"
    runP3(modelPath: modelPath, goldensPath: gp)  // exits
}
if args.contains("--p4") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard let modelPath = pos.first else {
        fail("--p4 needs <model.safetensors> [p4_goldens.safetensors]")
    }
    let gp = pos.dropFirst().first ?? "Goldens/p4_goldens.safetensors"
    runP4(modelPath: modelPath, goldensPath: gp)  // exits
}
if args.contains("--p5") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard let modelPath = pos.first else {
        fail("--p5 needs <model.safetensors> [p5_goldens.safetensors]")
    }
    let gp = pos.dropFirst().first ?? "Goldens/p5_goldens.safetensors"
    runP5(modelPath: modelPath, goldensPath: gp)  // exits
}
if args.contains("--p6") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard let modelPath = pos.first else {
        fail("--p6 needs <model.safetensors> [p6_goldens.safetensors]")
    }
    let gp = pos.dropFirst().first ?? "Goldens/p6_goldens.safetensors"
    runP6(modelPath: modelPath, goldensPath: gp)  // exits
}

if args.contains("--p7") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard let codecPath = pos.first else {
        fail("--p7 needs <codec.safetensors> [p7_goldens.safetensors]")
    }
    let gp = pos.dropFirst().first ?? "Goldens/p7_goldens.safetensors"
    runP7(codecPath: codecPath, goldensPath: gp)  // exits
}
if args.contains("--p8") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard pos.count >= 2 else {
        fail("--p8 needs <model.safetensors> <codec.safetensors> [p8_goldens.safetensors]")
    }
    let gp = pos.count >= 3 ? pos[2] : "Goldens/p8_goldens.safetensors"
    runP8(modelPath: pos[0], codecPath: pos[1], goldensPath: gp)  // exits
}
if args.contains("--p9") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard let codecPath = pos.first else {
        fail("--p9 needs <codec.safetensors> [p9_goldens.safetensors]")
    }
    let gp = pos.dropFirst().first ?? "Goldens/p9_goldens.safetensors"
    runP9(codecPath: codecPath, goldensPath: gp)  // exits
}
if args.contains("--p10") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    let tokFolder = pos.first
        ?? (NSHomeDirectory() + "/Development/_gepard-oracle/gepard_ckpt")
    let tokJson = pos.dropFirst().first ?? "Goldens/token_ids.json"
    runP10(tokenizerFolder: tokFolder, tokenGoldensPath: tokJson)  // exits
}
if args.contains("--ab-dtype") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    guard pos.count >= 2 else {
        fail("--ab-dtype needs <model.safetensors> <codec.safetensors> [p8_goldens.safetensors]")
    }
    let gp = pos.count >= 3 ? pos[2] : "Goldens/p8_goldens.safetensors"
    runABDtype(modelPath: pos[0], codecPath: pos[1], goldensPath: gp)  // exits
}
if args.contains("--validate") {
    let pos = args.dropFirst().filter { !$0.hasPrefix("--") }
    let modelDir = pos.first ?? (NSHomeDirectory() + "/Development/_gepard-oracle/gepard_ckpt")
    let codecDir = pos.dropFirst().first ?? (NSHomeDirectory() + "/Development/_gepard-oracle/codec_publish")
    let refWav = pos.dropFirst(2).first
        ?? (NSHomeDirectory() + "/Development/mlxengine-audio/WIP/gepard/oracle-capture/extra_ref_audio/roxy_a1.wav")
    runValidate(modelDir: modelDir, codecDir: codecDir, refWavPath: refWav)  // exits
}

guard args.contains("--p2") else {
    print("usage: gepard-gates --p2 [p2_goldens.json]")
    print("       gepard-gates --p3 <model.safetensors> [p3_goldens.safetensors]")
    print("       gepard-gates --p7 <codec.safetensors> [p7_goldens.safetensors]")
    exit(2)
}
let goldenPath = args.dropFirst().first(where: { !$0.hasPrefix("--") }) ?? "Goldens/p2_goldens.json"
guard let data = FileManager.default.contents(atPath: goldenPath) else {
    fail("goldens not found at \(goldenPath) (run from the package dir or pass a path)")
}
let g: P2Goldens
do { g = try JSONDecoder().decode(P2Goldens.self, from: data) }
catch { fail("decode goldens: \(error)") }

var problems = 0
func check(_ cond: Bool, _ msg: @autoclosure () -> String) { if !cond { problems += 1; print("  ✗ \(msg())") } }

// MARK: - P2a: TextRepetition

print("P2a TextRepetition  (\(g.textrep_cases.count) cases)")
let trc = TextRepetitionConfig(enabled: g.textrep_config.enabled,
                               targetTextTokens: g.textrep_config.target_text_tokens,
                               applyBelow: g.textrep_config.apply_below,
                               maxRepeats: g.textrep_config.max_repeats)
let rep = TextRepeater(config: trc, startOfText: g.special_tokens.sot,
                       endOfText: g.special_tokens.eot, startOfSpeech: g.special_tokens.sos)
for c in g.textrep_cases {
    let R = rep.targetR(c.ids.count)
    let out = rep.expand(c.ids)
    check(R == c.R, "\(c.name): R=\(R) golden=\(c.R)")
    check(out == c.cond_ids, "\(c.name): cond_ids mismatch (len \(out.count) vs \(c.cond_ids.count))")
}
print("  \(problems == 0 ? "✓" : "✗") \(g.textrep_cases.count) cases, tokens \(g.textrep_cases.reduce(0){$0+$1.cond_ids.count})")

// MARK: - P2b: CodecOps unfold  (packed [8][T] -> [32][T], compare vs ref_codes [T][32])

let p0 = problems
let unf = CodecOps.unfoldTokens(g.unfold.packed, levels: g.fsq_levels)   // [32][T]
let refTC = g.unfold.expected_ref_codes_TC                              // [T][32]
let T = unf.first?.count ?? 0
check(unf.count == g.fsq_levels.count * g.unfold.packed.count, "unfold channel count \(unf.count)")
var unfoldMismatch = 0
for ch in 0..<unf.count { for t in 0..<T where unf[ch][t] != refTC[t][ch] { unfoldMismatch += 1 } }
check(unfoldMismatch == 0, "unfold: \(unfoldMismatch) integer mismatches")
print("P2b CodecOps.unfold   [\(g.unfold.packed.count)x\(T)] -> [\(unf.count)x\(T)]: \(problems == p0 ? "✓ integer-exact" : "✗")")

// MARK: - P2c: CodecOps dequantize  ([32][T] -> [32][T] float, compare vs [T][32])

let p1 = problems
let deq = CodecOps.dequantize(g.dequant.codes, levels: g.fsq_levels, numLayers: g.num_layers)  // [32][T]
let deqTC = g.dequant.expected_TC                                                              // [T][32]
let T2 = deq.first?.count ?? 0
var maxAbs = 0.0
for ch in 0..<deq.count { for t in 0..<T2 { maxAbs = max(maxAbs, abs(Double(deq[ch][t]) - deqTC[t][ch])) } }
check(maxAbs < 1e-6, "dequantize: max|diff|=\(maxAbs)")
print("P2c CodecOps.dequant  [\(deq.count)x\(T2)]: max|diff|=\(String(format: "%.2e", maxAbs)) \(problems == p1 ? "✓" : "✗")")

print("")
if problems == 0 { print("P2 PASS — CodecOps + TextRepetition bit-exact vs goldens"); exit(0) }
else { print("P2 FAIL — \(problems) problem(s)"); exit(1) }

// MARK: - P3: Backbone prefill parity (MLX, CPU stream)

func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func relDiff(_ a: MLXArray, _ g: MLXArray) -> Float {
    let absmax = abs(g.asType(.float32)).max().item(Float.self)
    return maxAbsDiff(a, g) / max(absmax, 1e-6)
}

func runP3(modelPath: String, goldensPath: String) -> Never {
    // Torch goldens were CPU fp32; pin the CPU stream so deep-stack fp32 accumulation
    // matches instead of drifting on GPU. (mlx-porting doctrine.)
    Device.setDefault(device: Device(.cpu))

    print("P3 GepardBackbone prefill parity  (CPU stream)")

    let mURL = URL(fileURLWithPath: modelPath)
    let gURL = URL(fileURLWithPath: goldensPath)

    // --- weights: keep only model.* , strip prefix, cast bf16 -> fp32 ---
    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: mURL) } catch { fail("load model: \(error)") }
    var stripped: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("model.") {
        stripped[String(k.dropFirst("model.".count))] = v.asType(.float32)
    }
    guard stripped["embed_tokens.weight"] != nil, stripped["norm.weight"] != nil else {
        fail("model.* backbone keys missing (got \(stripped.count) stripped keys)")
    }
    print("  loaded \(stripped.count) backbone tensors (of \(raw.count))")

    let cfg = GepardBackboneConfig()
    let model = GepardBackbone(cfg)
    do {
        let params = ModuleParameters.unflattened(stripped)
        try model.update(parameters: params, verify: [.all])
    } catch { fail("weight bind: \(error)") }
    eval(model)

    // --- goldens ---
    let g: [String: MLXArray]
    do { g = try loadArrays(url: gURL) } catch { fail("load goldens: \(error)") }
    guard let prefixOut = g["prefix_out"]?.asType(.float32),      // [1,8,1024]
          let condIds = g["cond_ids"],                            // [T_text] int
          let layer0G = g["layer0_out"]?.asType(.float32),
          let layerLastG = g["layer_last_out"]?.asType(.float32),
          let lastHiddenG = g["last_hidden"]?.asType(.float32)
    else { fail("p3 goldens missing a tensor") }

    // inputsEmbeds = [ Q-Former speaker prefix (K=8) | embed(cond_ids) ]
    let ids2d = condIds.asType(.int32).reshaped(1, -1)            // [1,T]
    let textEmbeds = model.embed(ids2d)                          // [1,T,1024]
    let inputs = concatenated([prefixOut, textEmbeds], axis: 1)   // [1,8+T,1024]
    eval(inputs)

    let cache = model.newCache()
    let (lastHidden, caps) = model(inputsEmbeds: inputs, cache: cache, capture: true)
    eval(lastHidden)
    eval(caps)

    // Relative tolerance: the goldens are torch-CPU fp32, but our MLX path uses fused SDPA +
    // MLXFast.rmsNorm vs eager torch, a per-layer kernel difference of ~5e-4 (not accumulation:
    // it barely grows with depth). Structural correctness is proven by the exact std/absmax
    // match at layer0. argmax token selection (P6) is robust to this. Threshold 2e-3 relative.
    let TOL: Float = 2e-3
    func rel(_ a: MLXArray, _ g: MLXArray) -> Float {
        let absmax = abs(g.asType(.float32)).max().item(Float.self)
        return maxAbsDiff(a, g) / max(absmax, 1e-6)
    }
    let r0 = rel(caps[0], layer0G)
    let rL = rel(caps[caps.count - 1], layerLastG)
    let rH = rel(lastHidden, lastHiddenG)

    print("  seq len \(inputs.dim(1)) (K=\(prefixOut.dim(1)) speaker prefix + text \(textEmbeds.dim(1)))")
    print(String(format: "  layer0      rel|Δ| = %.2e  (abs %.2e)  %@", r0, maxAbsDiff(caps[0], layer0G), r0 < TOL ? "✓" : "✗"))
    print(String(format: "  layer_last  rel|Δ| = %.2e  (abs %.2e)  %@", rL, maxAbsDiff(caps[caps.count - 1], layerLastG), rL < TOL ? "✓" : "✗"))
    print(String(format: "  last_hidden rel|Δ| = %.2e  (abs %.2e)  %@  <- feeds the heads", rH, maxAbsDiff(lastHidden, lastHiddenG), rH < TOL ? "✓" : "✗"))
    print("")
    if r0 < TOL && rL < TOL && rH < TOL {
        print("P3 PASS — GepardBackbone prefill matches torch-CPU fp32 goldens (rel < \(TOL))"); exit(0)
    } else {
        print("P3 FAIL — backbone prefill drift exceeds tolerance"); exit(1)
    }
}

// MARK: - P4: Audio interface + codebook/stop heads parity (MLX, CPU stream)

func runP4(modelPath: String, goldensPath: String) -> Never {
    Device.setDefault(device: Device(.cpu))
    print("P4 AudioInterface + heads parity  (CPU stream)")

    let mURL = URL(fileURLWithPath: modelPath)
    let gURL = URL(fileURLWithPath: goldensPath)
    let levels = (0 ..< 32).map { [8, 7, 6, 6][$0 % 4] }

    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: mURL) } catch { fail("load model: \(error)") }

    // --- audio interface ---
    let ai = AudioInterface(levels: levels)
    var aiW: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("audio_embeddings.") || k.hasPrefix("audio_embed_proj.") || k == "audio_embed_scale" {
        aiW[AudioInterface.remapProjKey(k)] = v.asType(.float32)
    }
    do { try ai.update(parameters: ModuleParameters.unflattened(aiW), verify: [.all]) }
    catch { fail("audio-interface bind: \(error)") }

    // --- heads ---
    let ch = CodebookHeads(levels: levels)
    var chW: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("codebook_heads.") || k.hasPrefix("stop_head.") {
        chW[k] = v.asType(.float32)
    }
    do { try ch.update(parameters: ModuleParameters.unflattened(chW), verify: [.all]) }
    catch { fail("heads bind: \(error)") }
    eval(ai, ch)
    print("  bound \(aiW.count) interface + \(chW.count) head tensors")

    let g: [String: MLXArray]
    do { g = try loadArrays(url: gURL) } catch { fail("load goldens: \(error)") }
    let TOL: Float = 2e-3

    // P4b: audio interface (32 codes -> frame embed)
    guard let tokens = g["fixed_frame_tokens"], let feG = g["fixed_frame_embed"]?.asType(.float32)
    else { fail("p4 interface goldens missing") }
    let codes = tokens.asType(.int32).asArray(Int32.self).map { Int($0) }
    let fe = ai.frameEmbed(codes: codes)                       // [1,1,1024]
    let feRel = relDiff(fe, feG)
    print(String(format: "  frame_embed  rel|Δ| = %.2e  (abs %.2e)  %@", feRel, maxAbsDiff(fe, feG), feRel < TOL ? "✓" : "✗"))

    // P4a: heads on last_hidden's final position
    guard let lh = g["last_hidden"]?.asType(.float32), let stopG = g["stop_logit"]?.asType(.float32)
    else { fail("p4 head goldens missing") }
    let last = lh[0..., (lh.dim(1) - 1)..., 0...]              // [1,1,1024]
    let (logits, stop) = ch(last)
    var worstHead: Float = 0
    var worstIdx = 0
    for i in 0 ..< 32 {
        guard let gi = g[String(format: "head_%02d", i)]?.asType(.float32) else { fail("missing head_\(i)") }
        let r = relDiff(logits[i].reshaped(gi.shape), gi)
        if r > worstHead { worstHead = r; worstIdx = i }
    }
    let stopRel = relDiff(stop.reshaped(stopG.shape), stopG)
    print(String(format: "  32 heads     rel|Δ| = %.2e  (worst = head_%02d)  %@", worstHead, worstIdx, worstHead < TOL ? "✓" : "✗"))
    print(String(format: "  stop_head    rel|Δ| = %.2e  (abs %.2e)  %@", stopRel, maxAbsDiff(stop.reshaped(stopG.shape), stopG), stopRel < TOL ? "✓" : "✗"))
    print("")
    if feRel < TOL && worstHead < TOL && stopRel < TOL {
        print("P4 PASS — audio interface + heads match torch-CPU fp32 goldens (rel < \(TOL))"); exit(0)
    } else {
        print("P4 FAIL — parity exceeds tolerance"); exit(1)
    }
}

// MARK: - P5: Q-Former (RefCompressor) ladder parity (MLX, CPU stream)

func runP5(modelPath: String, goldensPath: String) -> Never {
    Device.setDefault(device: Device(.cpu))
    print("P5 RefCompressor (Q-Former) parity  (CPU stream)")

    let mURL = URL(fileURLWithPath: modelPath)
    let gURL = URL(fileURLWithPath: goldensPath)

    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: mURL) } catch { fail("load model: \(error)") }
    var rcW: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("ref_compressor.") {
        rcW[String(k.dropFirst("ref_compressor.".count))] = v.asType(.float32)
    }
    guard rcW["queries"] != nil, rcW["output_scale"] != nil else {
        fail("ref_compressor.* keys missing (got \(rcW.count))")
    }
    let rc = RefCompressor()
    do { try rc.update(parameters: ModuleParameters.unflattened(rcW), verify: [.all]) }
    catch { fail("ref_compressor bind: \(error)") }
    eval(rc)
    print("  bound \(rcW.count) ref_compressor tensors")

    let g: [String: MLXArray]
    do { g = try loadArrays(url: gURL) } catch { fail("load goldens: \(error)") }
    guard let refCodes = g["ref_codes"] else { fail("p5 goldens missing ref_codes") }

    let lad = rc.ladder(refCodes: refCodes.asType(.int32))
    eval(lad.prefix, lad.qNormed)

    let TOL: Float = 2e-3
    func line(_ name: String, _ a: MLXArray, _ key: String) -> Bool {
        guard let gg = g[key]?.asType(.float32) else { fail("missing golden \(key)") }
        let r = relDiff(a, gg)
        print(String(format: "  %-13@ rel|Δ| = %.2e  (abs %.2e)  %@", name, r, maxAbsDiff(a, gg), r < TOL ? "✓" : "✗"))
        return r < TOL
    }
    var ok = true
    ok = line("input_proj", lad.inputProjOut, "input_proj_out") && ok
    ok = line("pos_enc", lad.posEncOut, "pos_enc_out") && ok
    ok = line("block0", lad.blockOuts[0], "block0_out") && ok
    ok = line("block1", lad.blockOuts[1], "block1_out") && ok
    ok = line("q_normed", lad.qNormed, "q_normed") && ok
    ok = line("prefix_out", lad.prefix, "prefix_out") && ok
    print("")
    if ok { print("P5 PASS — Q-Former ladder matches torch-CPU fp32 goldens (rel < \(TOL))"); exit(0) }
    else { print("P5 FAIL — parity exceeds tolerance"); exit(1) }
}

// MARK: - P6: Greedy end-to-end rollout parity (MLX, CPU stream)

// Build+bind the three decoder modules from a checkpoint dict (reuses P3/P4 conventions).
func loadBackbone(_ raw: [String: MLXArray]) -> GepardBackbone {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("model.") { w[String(k.dropFirst(6))] = v.asType(.float32) }
    let m = GepardBackbone(GepardBackboneConfig())
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
    catch { fail("backbone bind: \(error)") }
    return m
}
func loadAudio(_ raw: [String: MLXArray], levels: [Int]) -> AudioInterface {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("audio_embeddings.") || k.hasPrefix("audio_embed_proj.") || k == "audio_embed_scale" {
        w[AudioInterface.remapProjKey(k)] = v.asType(.float32)
    }
    let m = AudioInterface(levels: levels)
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
    catch { fail("audio bind: \(error)") }
    return m
}
func loadHeads(_ raw: [String: MLXArray], levels: [Int]) -> CodebookHeads {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("codebook_heads.") || k.hasPrefix("stop_head.") { w[k] = v.asType(.float32) }
    let m = CodebookHeads(levels: levels)
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
    catch { fail("heads bind: \(error)") }
    return m
}

func runP6(modelPath: String, goldensPath: String) -> Never {
    Device.setDefault(device: Device(.cpu))
    print("P6 greedy rollout parity  (CPU stream)")

    let mURL = URL(fileURLWithPath: modelPath)
    let gURL = URL(fileURLWithPath: goldensPath)
    let levels = (0 ..< 32).map { [8, 7, 6, 6][$0 % 4] }

    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: mURL) } catch { fail("load model: \(error)") }
    let dec = GepardDecoder(backbone: loadBackbone(raw), audio: loadAudio(raw, levels: levels),
                            heads: loadHeads(raw, levels: levels))
    eval(dec.backbone, dec.audio, dec.heads)

    let g: [String: MLXArray]
    do { g = try loadArrays(url: gURL) } catch { fail("load goldens: \(error)") }
    guard let prefix = g["prefix"]?.asType(.float32) else { fail("p6 missing prefix") }
    let maxFrames = 120

    // A mismatch whose winning-vs-golden logit gap is below this is an fp knife-edge tie,
    // not a computational error (the fused SDPA/rmsNorm stack drifts ~5e-4; see P3).
    let TIE: Float = 5e-3
    var allPass = true
    for name in ["canonical", "one_word"] {
        guard let condIds = g["\(name)_cond_ids"],
              let goldCodesArr = g["\(name)_codes"]?.asType(.int32)
        else { fail("p6 missing \(name) tensors") }
        let Tg = goldCodesArr.dim(1)
        let goldFlat = goldCodesArr.asArray(Int32.self)             // [32*Tg], channel-major
        func goldFrame(_ t: Int) -> [Int] { (0 ..< 32).map { Int(goldFlat[$0 * Tg + t]) } }

        // (1) free-running greedy rollout — real behaviour, cascades on any flip
        let roll = dec.greedyRollout(prefix: prefix, condIds: condIds, maxFrames: maxFrames)
        let Tm = roll.codes.count
        var matchLen = 0
        while matchLen < min(Tm, Tg) {
            if (0 ..< 32).contains(where: { roll.codes[matchLen][$0] != goldFrame(matchLen)[$0] }) { break }
            matchLen += 1
        }

        // (2) teacher-forced — feed GOLDEN history so hidden states track the oracle exactly;
        //     isolates per-frame argmax parity from cascade. Records every codebook flip + gap.
        var (hidden, cache) = dec.prefill(prefix: prefix, condIds: condIds)
        var mismatches = 0, decisive = 0
        var worstTie: Float = 0
        func scoreFrame(_ t: Int, _ logits: [MLXArray]) {
            let gf = goldFrame(t)
            for c in 0 ..< 32 {
                let v = logits[c].reshaped(-1).asArray(Float.self)
                let mine = v.firstIndex(of: v.max()!)!
                if mine != gf[c] {
                    mismatches += 1
                    let gap = v[mine] - v[gf[c]]
                    if gap >= TIE { decisive += 1 } else { worstTie = max(worstTie, gap) }
                }
            }
        }
        scoreFrame(0, dec.heads(hidden).logits)
        for t in 1 ..< Tg {
            hidden = dec.decodeStep(frameEmbed: dec.audio.frameEmbed(codes: goldFrame(t - 1)), cache: cache)
            scoreFrame(t, dec.heads(hidden).logits)
        }

        let ok = (decisive == 0)
        let tag = matchLen == Tg && Tm == Tg ? "✓ integer-exact"
            : ok ? "✓ ties-only (fp knife-edge)" : "✗ decisive flip"
        print("  \(name): free-run \(Tm)f prefix-match \(matchLen)/\(Tg) · teacher-forced \(mismatches) flips (\(decisive) decisive, worst tie gap \(String(format: "%.1e", worstTie))/\(32 * Tg) codes)  \(tag)")
        if !ok { allPass = false }
    }
    print("")
    if allPass { print("P6 PASS — per-codebook argmax matches oracle up to fp knife-edge ties (< \(TIE))"); exit(0) }
    else { print("P6 FAIL — a decisive (non-tie) argmax flip indicates a real discrepancy"); exit(1) }
}

// MARK: - P7: NanoCodec decoder parity (32 code channels -> waveform, MLX, CPU stream)

func runP7(codecPath: String, goldensPath: String) -> Never {
    // Golden waveform is the NeMo torch-CPU-fp32 decode; pin the CPU stream so deep-conv
    // fp32 accumulation matches instead of drifting on the GPU (Phase-1 hit 6.7e-6 on CPU).
    Device.setDefault(device: Device(.cpu))
    print("P7 NanoCodec decoder parity  (CPU stream)")

    let cURL = URL(fileURLWithPath: codecPath)
    let gURL = URL(fileURLWithPath: goldensPath)

    // --- codec weights (full 984-tensor safetensors; the decoder pulls its ~387 keys) ---
    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: cURL) } catch { fail("load codec: \(error)") }
    let decKeys = raw.keys.filter { $0.hasPrefix("audio_decoder.") }.count
    guard decKeys > 0 else { fail("no audio_decoder.* keys in \(codecPath) (got \(raw.count) tensors)") }
    let dec = NanoCodecDecoder(weights: raw)
    print("  built decoder from \(decKeys) audio_decoder.* tensors (of \(raw.count))")

    // --- goldens ---
    let g: [String: MLXArray]
    do { g = try loadArrays(url: gURL) } catch { fail("load goldens: \(error)") }
    guard let decIn = g["decoder_input"]?.asType(.float32),      // [1, 32, T]
          let waveG = g["decoded_wave"]?.asType(.float32)        // [T*1024]
    else { fail("p7 goldens missing decoder_input / decoded_wave") }

    // --- decode ---
    let out = dec.decode(decIn)                                  // [1, 1, T*1024]
    eval(out)
    let wave = out.reshaped([-1])                                // [T*1024]

    // --- compare ---
    let n = min(wave.dim(0), waveG.dim(0))
    let a = wave[0 ..< n], b = waveG[0 ..< n]
    let diff = a - b
    let maxAbs = abs(diff).max().item(Float.self)
    let rms = sqrt((diff * diff).mean()).item(Float.self)
    // scale-invariant SNR (dB): 10 log10( ||b||^2 / ||a-b||^2 )
    let sigE = (b * b).sum().item(Float.self)
    let errE = (diff * diff).sum().item(Float.self)
    let siSNR = 10.0 * Foundation.log10(Double(sigE) / (Double(errE) + 1e-12))
    // Pearson correlation
    let am = a.mean(), bm = b.mean()
    let ac = a - am, bc = b - bm
    let corr = (ac * bc).sum().item(Float.self)
        / (sqrt((ac * ac).sum()).item(Float.self) * sqrt((bc * bc).sum()).item(Float.self) + 1e-12)

    print("  wave len \(wave.dim(0))  golden len \(waveG.dim(0))  compared n=\(n)")
    print(String(format: "  max|diff| = %.4e   rms = %.4e", maxAbs, rms))
    print(String(format: "  corr = %.6f   SI-SNR = %.1f dB", corr, siSNR))
    print("")

    // Phase-1 (MLX-Python, same CPU stream) hit max_abs 6.7e-6 / SI-SNR 104 dB. Generous gate:
    // max abs < 1e-3 (expect ~1e-5); require high corr + SI-SNR as a shape/scale sanity check.
    let pass = maxAbs < 1e-3 && corr > 0.999 && siSNR > 40
    if pass {
        print("P7 PASS — NanoCodec decoder matches NeMo torch-CPU-fp32 waveform (max|Δ| < 1e-3)"); exit(0)
    } else {
        print("P7 FAIL — decoder output diverges from the golden waveform"); exit(1)
    }
}

// MARK: - P8: full-pipeline GPU smoke (Metal / GPU stream) — audible-valid + RTF/TTFA

func loadRefCompressor(_ raw: [String: MLXArray]) -> RefCompressor {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("ref_compressor.") { w[String(k.dropFirst("ref_compressor.".count))] = v.asType(.float32) }
    let m = RefCompressor()
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) } catch { fail("ref_compressor bind: \(error)") }
    return m
}

func writeWav16(_ samples: [Float], to path: String, sampleRate: Int = 22050) {
    var d = Data()
    func s16(_ v: Int16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    let n = samples.count
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + n * 2)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); s16(1); s16(1)
    u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); s16(2); s16(16)
    d.append("data".data(using: .ascii)!); u32(UInt32(n * 2))
    for x in samples { s16(Int16(max(-1, min(1, x)) * 32767)) }
    try? d.write(to: URL(fileURLWithPath: path))
}

func runP8(modelPath: String, codecPath: String, goldensPath: String) -> Never {
    Device.setDefault(device: Device(.gpu))       // real runtime path — GPU/Metal, NOT CPU-pinned
    print("P8 full-pipeline GPU smoke  (Metal / GPU stream)")
    func ms() -> Double { Date().timeIntervalSince1970 * 1000 }

    let levels = (0 ..< 32).map { [8, 7, 6, 6][$0 % 4] }
    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: URL(fileURLWithPath: modelPath)) } catch { fail("load model: \(error)") }
    let dec = GepardDecoder(backbone: loadBackbone(raw), audio: loadAudio(raw, levels: levels), heads: loadHeads(raw, levels: levels))
    let rc = loadRefCompressor(raw)
    let craw: [String: MLXArray]
    do { craw = try loadArrays(url: URL(fileURLWithPath: codecPath)) } catch { fail("load codec: \(error)") }
    let codec = NanoCodecDecoder(weights: craw)
    eval(dec.backbone, dec.audio, dec.heads, rc)
    print("  model + codec loaded")

    let g: [String: MLXArray]
    do { g = try loadArrays(url: URL(fileURLWithPath: goldensPath)) } catch { fail("load goldens: \(error)") }
    guard let refCodes = g["ref_codes"]?.asType(.int32), let condIds = g["canonical_cond_ids"],
          let decIn = g["decoder_input"]?.asType(.float32), let waveG = g["decoded_wave"]?.asType(.float32)
    else { fail("p8 goldens incomplete") }

    // (A) decoder correctness on GPU vs the CPU-fp32 golden
    let outA = codec.decode(decIn); eval(outA)
    let waveA = outA.reshaped([-1])
    let nA = min(waveA.dim(0), waveG.dim(0))
    let dA = waveA[0 ..< nA] - waveG[0 ..< nA]
    let corrA: Float = {
        let a = waveA[0 ..< nA] - waveA[0 ..< nA].mean(), b = waveG[0 ..< nA] - waveG[0 ..< nA].mean()
        return (a * b).sum().item(Float.self) / (sqrt((a * a).sum() * (b * b).sum()).item(Float.self) + 1e-12)
    }()
    print(String(format: "  (A) decoder on GPU vs CPU golden: max|Δ| %.2e  corr %.6f", abs(dA).max().item(Float.self), corrA))

    // warmup — force first-forward Metal kernel compilation off the clock
    let (wh, wc) = dec.prefill(prefix: rc(refCodes: refCodes).prefix, condIds: condIds)
    _ = dec.heads(wh).logits.map { $0.eval() }; _ = wc

    // (B) end-to-end timing: ref-encode → prefill+frame0 (TTFA) → full rollout → decode
    let t0 = ms()
    let prefix = rc(refCodes: refCodes).prefix; eval(prefix)
    let tRef = ms()
    let (h0, _) = dec.prefill(prefix: prefix, condIds: condIds)
    let f0 = dec.argmaxFrame(logits: dec.heads(h0).logits)                 // forces eval
    let tPre = ms()
    func codesToInput(_ frames: [[Int]]) -> MLXArray {
        let T = frames.count
        var flat = [Int32](repeating: 0, count: 32 * T)
        for t in 0 ..< T { for c in 0 ..< 32 { flat[c * T + t] = Int32(frames[t][c]) } }
        return NanoCodecDecoder.dequantize(MLXArray(flat, [1, 32, T]))
    }
    let firstDec = codec.decode(codesToInput([f0])); eval(firstDec)
    let tFirst = ms()

    let roll = dec.greedyRollout(prefix: prefix, condIds: condIds, maxFrames: 200)
    let tRoll = ms()
    let frames = roll.codes.count
    let wave = codec.decode(codesToInput(roll.codes)).reshaped([-1]); eval(wave)
    let tDec = ms()

    let samples = wave.asArray(Float.self)
    let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count))
    let peak = samples.map { abs($0) }.max() ?? 0
    let finite = samples.allSatisfy { $0.isFinite }
    let audioSec = Double(frames * 1024) / 22050.0
    let genSec = (tRoll - tRef) / 1000.0
    let e2eSec = (tDec - t0) / 1000.0
    let ttfaMs = (tRef - t0) + (tPre - tRef) + (tFirst - tPre)   // ref-enc + prefill+frame0 + 1-frame decode

    let outWav = "/private/tmp/claude-501/-Users-dustinnielson-Development-MLXEngine/0c989819-ef4f-47cc-aeb1-57912ea98bf5/scratchpad/p8_roxy_a1_canonical.wav"
    writeWav16(samples, to: outWav)

    print("  (B) end-to-end (roxy_a1 ref + canonical text):")
    print(String(format: "      ref-encode %.0f ms · prefill+frame0 %.0f ms · rollout %d frames %.0f ms (%.1f ms/frame) · decode %.0f ms",
                 tRef - t0, tPre - tRef, frames, tRoll - tPre, (tRoll - tPre) / Double(max(frames - 1, 1)), tDec - tRoll))
    print(String(format: "      audio %.2fs · gen-RTF %.3f · e2e-RTF %.3f · TTFA ≈ %.0f ms · %.1f× realtime",
                 audioSec, genSec / audioSec, e2eSec / audioSec, ttfaMs, audioSec / e2eSec))
    print(String(format: "      wave: %d samples · rms %.4f · peak %.3f · finite %@", samples.count, rms, peak, finite ? "yes" : "NO"))
    print("      wrote \(outWav)")
    print("")
    let audible = finite && rms > 1e-3 && peak > 0.02 && samples.count == frames * 1024
    if audible { print("P8 PASS — full pipeline runs on GPU, audio is valid (finite, non-silent, correct length)"); exit(0) }
    else { print("P8 FAIL — output not audible-valid (finite=\(finite) rms=\(rms) peak=\(peak))"); exit(1) }
}

// MARK: - P9: NanoCodec encoder parity (waveform -> packed tokens; CPU stream)

func runP9(codecPath: String, goldensPath: String) -> Never {
    Device.setDefault(device: Device(.cpu))   // match the CPU-fp32 Phase-1 reference / NeMo goldens
    print("P9 NanoCodec encoder parity  (CPU stream)")

    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: URL(fileURLWithPath: codecPath)) } catch { fail("load codec: \(error)") }
    let encKeys = raw.keys.filter { $0.hasPrefix("audio_encoder.") }.count
    guard encKeys > 0 else { fail("no audio_encoder.* keys in \(codecPath)") }
    let enc = NanoCodecEncoder(weights: raw)
    print("  built encoder from \(encKeys) audio_encoder.* tensors")

    let g: [String: MLXArray]
    do { g = try loadArrays(url: URL(fileURLWithPath: goldensPath)) } catch { fail("load goldens: \(error)") }

    let levels = [8, 7, 6, 6]
    var allPass = true
    for clip in ["roxy_a1", "roxy_a2"] {
        guard let wave = g["\(clip)_wave"]?.asType(.float32),
              let goldPacked = g["\(clip)_packed"]?.asType(.int32) else {
            print("  \(clip): goldens missing — skip"); continue
        }
        let packed = enc.encode(wave.reshaped([1, -1]))   // [1,8,T]
        eval(packed)
        let Tg = goldPacked.dim(2), Tm = packed.dim(2)
        let T = min(Tg, Tm)

        // integer-exact comparison on the 8 packed group indices (+ unfolded 32-ch view)
        let a = packed[0..., 0..., 0 ..< T].asArray(Int32.self)          // [8*T] row-major
        let b = goldPacked[0..., 0..., 0 ..< T].asArray(Int32.self)
        var exact = 0
        for i in 0 ..< a.count where a[i] == b[i] { exact += 1 }
        let pctPacked = 100.0 * Double(exact) / Double(a.count)

        // unfolded [32,T] via the P2-validated CodecOps (the Q-Former's actual input)
        let packed2D = (0 ..< 8).map { grp in (0 ..< T).map { t in Int(b[grp * T + t]) } }  // golden [8][T]
        let mine2D = (0 ..< 8).map { grp in (0 ..< T).map { t in Int(a[grp * T + t]) } }
        let unfMine = CodecOps.unfoldTokens(mine2D, levels: levels)      // [32][T]
        let unfGold = CodecOps.unfoldTokens(packed2D, levels: levels)
        var uExact = 0, uTot = 0
        for ch in 0 ..< unfMine.count { for t in 0 ..< T { uTot += 1; if unfMine[ch][t] == unfGold[ch][t] { uExact += 1 } } }
        let pctUnf = 100.0 * Double(uExact) / Double(uTot)

        // Accept the documented Phase-1 tail residual: ≤1 frame length diff (conv edge effect;
        // the Q-Former masks ref codes so a tail frame is immaterial) + FSQ boundary flips.
        let ok = pctPacked >= 99.0 && abs(Tm - Tg) <= 1
        let tail = Tm == Tg ? "" : "  (Δframes \(Tm - Tg) = accepted tail residual)"
        print(String(format: "  %@: frames %d (golden %d) · packed %.2f%% int-exact (%d/%d) · unfolded32 %.2f%%  %@%@",
                     clip, Tm, Tg, pctPacked, exact, a.count, pctUnf, ok ? "✓" : "✗", tail))
        if !ok { allPass = false }
    }
    print("")
    if allPass { print("P9 PASS — encoder codes ≥99% integer-exact vs NeMo goldens (FSQ boundary flips + ≤1 tail frame accepted)"); exit(0) }
    else { print("P9 FAIL — encoder codes below tolerance"); exit(1) }
}

// MARK: - P10: tokenizer → cond_ids (integer-exact vs the Stage-0 goldens)

// The last functional gap in run(): text → cond_ids. Loads the Qwen3.5 tokenizer.json faithfully
// (swift-transformers PreTrainedTokenizer), encodes each Stage-0 text (add_special_tokens=False),
// wraps with the TextRepeater (SOT/EOT/SOS + repetition), and asserts BOTH ids and cond_ids are
// integer-exact vs token_ids.json — then cross-checks the cond_ids embedded in the p3/p6
// safetensors goldens. A single-token drift here silently collapses WER, so this gate is
// integer-exact (no tolerance), the mlx-porting parity doctrine for the tokenizer sub-gate.

struct TokenGoldens: Codable {
    struct Case: Codable { let text: String; let ids: [Int]; let R: Int; let cond_ids: [Int] }
}

func runP10(tokenizerFolder: String, tokenGoldensPath: String) -> Never {
    Device.setDefault(device: Device(.cpu))
    print("P10 tokenizer → cond_ids parity  (integer-exact)")
    print("  tokenizer folder: \(tokenizerFolder)")

    // token_ids.json is a dict of name -> {text, ids, R, cond_ids}
    guard let data = FileManager.default.contents(atPath: tokenGoldensPath) else {
        fail("token goldens not found at \(tokenGoldensPath)")
    }
    let cases: [String: TokenGoldens.Case]
    do { cases = try JSONDecoder().decode([String: TokenGoldens.Case].self, from: data) }
    catch { fail("decode token_ids.json: \(error)") }

    // Load the conditioner (async → sync bridge for the CLI).
    let folder = URL(fileURLWithPath: tokenizerFolder, isDirectory: true)
    var conditioner: GepardConditioner?
    var loadErr: Error?
    let sem = DispatchSemaphore(value: 0)
    Task {
        do { conditioner = try await GepardConditioner.load(modelFolder: folder) }
        catch { loadErr = error }
        sem.signal()
    }
    sem.wait()
    if let loadErr { fail("tokenizer load: \(loadErr)") }
    guard let cond = conditioner else { fail("tokenizer load returned nil") }

    // Confirm the fixed special ids match gepard_config.json (defensive).
    guard GepardConditioner.sot == 248073, GepardConditioner.eot == 248074,
          GepardConditioner.sos == 248070 else { fail("special ids drifted from gepard_config.json") }

    var problems = 0
    let names = cases.keys.sorted()
    for name in names {
        let c = cases[name]!
        let ids = cond.encode(c.text)
        let R = cond.repeater.targetR(ids.count)
        let condIds = cond.condIds(for: c.text)
        let idsOK = ids == c.ids
        let rOK = R == c.R
        let condOK = condIds == c.cond_ids
        if !idsOK || !rOK || !condOK { problems += 1 }
        let tag = (idsOK && rOK && condOK) ? "✓" : "✗"
        print("  \(tag) \(name.padding(toLength: 16, withPad: " ", startingAt: 0)) "
              + "ids \(ids.count)\(idsOK ? "" : "≠\(c.ids.count)") · R \(R)\(rOK ? "" : "≠\(c.R)") "
              + "· cond_ids \(condIds.count)\(condOK ? "" : "≠\(c.cond_ids.count)")")
        if !idsOK { print("        ids mismatch: got \(ids.prefix(8))… golden \(c.ids.prefix(8))…") }
        if !condOK && idsOK { print("        cond_ids framing mismatch (ids matched — check TextRepeater)") }
    }

    // Cross-check the cond_ids embedded in the p3/p6 safetensors goldens (the same arrays run()
    // will actually feed prefill), so P10 also validates the tensor-golden path end-to-end.
    func crossCheck(_ safetensors: String, key: String, text: String, label: String) {
        guard FileManager.default.fileExists(atPath: safetensors) else {
            print("  (skip \(label): \(safetensors) not present)"); return
        }
        guard let g = try? loadArrays(url: URL(fileURLWithPath: safetensors)),
              let arr = g[key] else { print("  (skip \(label): key \(key) missing)"); return }
        let golden = arr.asType(.int32).asArray(Int32.self).map { Int($0) }
        let mine = cond.condIds(for: text)
        let ok = mine == golden
        if !ok { problems += 1 }
        print("  \(ok ? "✓" : "✗") cross-check \(label): cond_ids \(mine.count) vs safetensors \(golden.count)")
    }
    let canonicalText = cases["canonical"]?.text ?? ""
    let oneWordText = cases["one_word"]?.text ?? ""
    crossCheck("Goldens/p3_goldens.safetensors", key: "cond_ids", text: canonicalText, label: "p3/canonical")
    crossCheck("Goldens/p6_goldens.safetensors", key: "canonical_cond_ids", text: canonicalText, label: "p6/canonical")
    crossCheck("Goldens/p6_goldens.safetensors", key: "one_word_cond_ids", text: oneWordText, label: "p6/one_word")

    print("")
    if problems == 0 {
        print("P10 PASS — tokenizer → cond_ids integer-exact vs goldens (\(names.count) texts + 3 cross-checks)")
        exit(0)
    } else {
        print("P10 FAIL — \(problems) mismatch(es); a token drift collapses WER")
        exit(1)
    }
}

// MARK: - --validate: headless engine harness (footprint + dBFS + RTF/TTFA + short-word matrix)

/// Peak-normalized RMS in dBFS (non-silence probe; a silent stem reads −∞).
func dbfs(_ samples: [Float]) -> Double {
    guard !samples.isEmpty else { return -.infinity }
    let sumSq = samples.reduce(0.0) { $0 + Double($1) * Double($1) }
    let rms = (sumSq / Double(samples.count)).squareRoot()
    return rms > 0 ? 20.0 * Foundation.log10(rms) : -.infinity
}

/// Real-process resident memory (mach phys_footprint) in MB — the app-truth footprint (the
/// skill's note: MLX GPU peak under-reads ~2.7× vs phys).
func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576.0 : -1
}

func nowMs() -> Double { Date().timeIntervalSince1970 * 1000 }

func runValidate(modelDir: String, codecDir: String, refWavPath: String) -> Never {
    Device.setDefault(device: Device(.gpu))
    print("GEPARD_VALIDATE — engine run() harness (GPU/Metal)")
    print("  model: \(modelDir)")
    print("  codec: \(codecDir)")
    print("  ref:   \(refWavPath)")

    guard let refData = FileManager.default.contents(atPath: refWavPath) else {
        fail("reference wav not found at \(refWavPath)")
    }
    let refAudio = Audio(format: .wav, data: refData, sampleRate: nil, channels: 1)

    let config = GepardConfiguration(
        modelDirectory: URL(fileURLWithPath: modelDir, isDirectory: true),
        codecDirectory: URL(fileURLWithPath: codecDir, isDirectory: true))

    // Bound the MLX buffer-pool cache the way the engine (≥0.21.0) does, so phys_footprint and
    // the peak reading reflect the working set, not the reclaimable free-buffer ratchet.
    MLX.Memory.cacheLimit = 256 * 1024 * 1024

    let sem = DispatchSemaphore(value: 0)
    var failure: String?
    Task { @InferenceActor in
        defer { sem.signal() }
        let package = GepardPackage(configuration: config)

        // --- load() → resident floor (MLX active bytes = live working set) ---
        let physBefore = physFootprintMB()
        let tLoad0 = nowMs()
        do { try await package.load() } catch { failure = "load: \(error)"; return }
        let loadMs = nowMs() - tLoad0
        let residentMB = Double(MLX.Memory.activeMemory) / 1_048_576.0
        let physAfterLoad = physFootprintMB()
        print(String(format: "\n[LOAD] %.2f s · resident floor: MLX-active %.0f MB · phys %.0f MB (Δphys %.0f MB over %.0f MB baseline)",
                     loadMs / 1000, residentMB, physAfterLoad, physAfterLoad - physBefore, physBefore))

        // --- one full run() at the envelope (long paragraph) → RTF/TTFA/dBFS/peak ---
        func runOnce(text: String, cfg: Double?, maxFrames: Int, label: String,
                     writeWav: String? = nil) async -> (frames: Int, secs: Double, runMs: Double,
                                                          ttfaMs: Double, dbfs: Double, peakMB: Double, stopped: Bool)? {
            var meta: MetaData = ["maxFrames": .int(maxFrames)]
            if let cfg { meta["cfgScale"] = .double(cfg); meta["cfgFrames"] = .int(20) }
            let request = TTSRequest(
                text: text,
                voice: VoiceSelector(.referenceAudio(refAudio)),
                metaData: meta)
            var firstFrameMs: Double = 0
            MLX.Memory.peakMemory = 0     // reset high-water so peak reflects THIS run's activation
            let t0 = nowMs()
            let response: any CapabilityResponse
            do {
                response = try await RunProgress.$sink.withValue({ report in
                    if report.phase == .generate, report.step == 1, firstFrameMs == 0 {
                        firstFrameMs = nowMs() - t0
                    }
                }) {
                    try await package.run(request)
                }
            } catch { print("  [\(label)] run FAILED: \(error)"); return nil }
            let runMs = nowMs() - t0
            let peakMB = Double(MLX.Memory.peakMemory) / 1_048_576.0   // MLX high-water (live buffers)
            guard let tts = response as? TTSResponse else { print("  [\(label)] not a TTSResponse"); return nil }
            let samples = decodeWavSamples(tts.audio.data)
            let frames = samples.count / GepardModel.samplesPerFrame
            let secs = Double(samples.count) / Double(GepardModel.sampleRate)
            let db = dbfs(samples)
            let stopped = frames < maxFrames - 1
            if let writeWav { try? tts.audio.data.write(to: URL(fileURLWithPath: writeWav)) }
            print(String(format: "  [%@] %d frames · %.2f s audio · run %.0f ms · TTFA %.0f ms · RTF %.3f (%.1f× realtime) · %.1f dBFS · MLX-peak %.0f MB · %@",
                         label, frames, secs, runMs, firstFrameMs, runMs / 1000 / max(secs, 1e-6),
                         secs / (runMs / 1000), db, peakMB, stopped ? "STOPPED" : "hit cap"))
            return (frames, secs, runMs, firstFrameMs, db, peakMB, stopped)
        }

        let longText = "It's a beautiful morning. I already checked your calendar, and you have "
            + "two meetings before lunch. After that, the afternoon is completely free, so we could "
            + "finally finish the project we started last week."
        print("\n[ENVELOPE] long paragraph (warmup first):")
        _ = await runOnce(text: "Warmup.", cfg: nil, maxFrames: 60, label: "warmup")
        let scratch = NSHomeDirectory() + "/Development/_gepard-oracle/gepard_validate_paragraph.wav"
        _ = await runOnce(text: longText, cfg: nil, maxFrames: 2000, label: "paragraph", writeWav: scratch)

        // --- short-utterance matrix (deterministic greedy → one run per condition) ---
        print("\n[SHORT-WORD MATRIX] greedy is deterministic — one run/condition (CFG=onset text-CFG w=2.6):")
        for text in ["One.", "Okay!", "Hi."] {
            _ = await runOnce(text: text, cfg: nil, maxFrames: 200, label: "\(text) CFG-off")
            _ = await runOnce(text: text, cfg: 2.6, maxFrames: 200, label: "\(text) CFG-on ")
        }

        await package.unload()
        let physAfterUnload = physFootprintMB()
        print(String(format: "\n[UNLOAD] resident %.0f MB (released %.0f MB)",
                     physAfterUnload, physAfterLoad - physAfterUnload))
    }
    sem.wait()
    if let failure { fail(failure) }
    print("\nGEPARD_VALIDATE complete.")
    exit(0)
}

// MARK: - --ab-dtype: bf16 (runtime) vs fp32 (P-gate) rollout A/B on ONE GPU stream

// The P-gates validated the LM upcast to fp32 (CPU stream). The runtime (GepardModel) binds the
// LM at native bf16 for the ~1.1 GB resident target. This gate loads the SAME weights twice —
// once cast to fp32, once cast to bf16 — computes the speaker prefix + greedy rollout for each on
// the GPU stream (so DTYPE is the only variable, not device), and reports: (1) whether the bf16
// rollout is audible-valid after codec decode, (2) how far the two argmax code sequences agree,
// and (3) where/how they first diverge. Backbone GEMM K ≤ 3584 (< the NAX split-K bf16 hazard),
// so bf16 on this GPU is trustworthy. The codec stays fp32 in both arms (not the variable).

func bindBackboneDtype(_ raw: [String: MLXArray], _ dt: DType) -> GepardBackbone {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("model.") { w[String(k.dropFirst(6))] = v.asType(dt) }
    let m = GepardBackbone(GepardBackboneConfig())
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
    catch { fail("backbone bind (\(dt)): \(error)") }
    return m
}
func bindAudioDtype(_ raw: [String: MLXArray], levels: [Int], _ dt: DType) -> AudioInterface {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("audio_embeddings.") || k.hasPrefix("audio_embed_proj.") || k == "audio_embed_scale" {
        w[AudioInterface.remapProjKey(k)] = v.asType(dt)
    }
    let m = AudioInterface(levels: levels)
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
    catch { fail("audio bind (\(dt)): \(error)") }
    return m
}
func bindHeadsDtype(_ raw: [String: MLXArray], levels: [Int], _ dt: DType) -> CodebookHeads {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("codebook_heads.") || k.hasPrefix("stop_head.") { w[k] = v.asType(dt) }
    let m = CodebookHeads(levels: levels)
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
    catch { fail("heads bind (\(dt)): \(error)") }
    return m
}
func bindRefCompressorDtype(_ raw: [String: MLXArray], _ dt: DType) -> RefCompressor {
    var w: [String: MLXArray] = [:]
    for (k, v) in raw where k.hasPrefix("ref_compressor.") { w[String(k.dropFirst("ref_compressor.".count))] = v.asType(dt) }
    let m = RefCompressor()
    do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) } catch { fail("ref_compressor bind (\(dt)): \(error)") }
    return m
}

func runABDtype(modelPath: String, codecPath: String, goldensPath: String) -> Never {
    Device.setDefault(device: Device(.gpu))       // runtime device; dtype is the only variable
    print("AB-DTYPE  bf16 (runtime) vs fp32 (P-gate) rollout on the GPU stream")

    let levels = (0 ..< 32).map { [8, 7, 6, 6][$0 % 4] }
    let raw: [String: MLXArray]
    do { raw = try loadArrays(url: URL(fileURLWithPath: modelPath)) } catch { fail("load model: \(error)") }
    let craw: [String: MLXArray]
    do { craw = try loadArrays(url: URL(fileURLWithPath: codecPath)) } catch { fail("load codec: \(error)") }
    let codec = NanoCodecDecoder(weights: craw)   // fp32 in both arms — not the variable

    let g: [String: MLXArray]
    do { g = try loadArrays(url: URL(fileURLWithPath: goldensPath)) } catch { fail("load goldens: \(error)") }
    guard let refCodes = g["ref_codes"]?.asType(.int32), let condIds = g["canonical_cond_ids"]
    else { fail("ab goldens missing ref_codes / canonical_cond_ids") }

    let maxFrames = 200
    func rolloutFor(_ dt: DType) -> GepardDecoder.Rollout {
        let dec = GepardDecoder(backbone: bindBackboneDtype(raw, dt),
                                audio: bindAudioDtype(raw, levels: levels, dt),
                                heads: bindHeadsDtype(raw, levels: levels, dt))
        let rc = bindRefCompressorDtype(raw, dt)
        eval(dec.backbone, dec.audio, dec.heads, rc)
        let prefix = rc(refCodes: refCodes).prefix; eval(prefix)
        let r = dec.greedyRollout(prefix: prefix, condIds: condIds, maxFrames: maxFrames)
        return r
    }

    let fp32 = rolloutFor(.float32)
    let bf16 = rolloutFor(.bfloat16)
    print("  fp32: \(fp32.codes.count) frames · bf16: \(bf16.codes.count) frames")

    // (1) bf16 audible-valid after codec decode
    func decodeToSamples(_ frames: [[Int]]) -> [Float] {
        let T = frames.count
        var flat = [Int32](repeating: 0, count: 32 * T)
        for t in 0 ..< T { for c in 0 ..< 32 { flat[c * T + t] = Int32(frames[t][c]) } }
        let w = codec.decode(NanoCodecDecoder.dequantize(MLXArray(flat, [1, 32, T]))).reshaped([-1]); eval(w)
        return w.asArray(Float.self)
    }
    let sBf = decodeToSamples(bf16.codes)
    let rmsBf = sqrt(sBf.reduce(0) { $0 + $1 * $1 } / Float(max(sBf.count, 1)))
    let peakBf = sBf.map { abs($0) }.max() ?? 0
    let finiteBf = sBf.allSatisfy { $0.isFinite }
    let audibleBf = finiteBf && rmsBf > 1e-3 && peakBf > 0.02 && sBf.count == bf16.codes.count * 1024
    print(String(format: "  (1) bf16 audio: %d samples · rms %.4f · peak %.3f · finite %@ → %@",
                 sBf.count, rmsBf, peakBf, finiteBf ? "yes" : "NO", audibleBf ? "AUDIBLE-VALID" : "INVALID"))

    // (2)/(3) argmax agreement between the two dtype rollouts
    let common = min(fp32.codes.count, bf16.codes.count)
    var matchLen = 0
    while matchLen < common {
        if (0 ..< 32).contains(where: { fp32.codes[matchLen][$0] != bf16.codes[matchLen][$0] }) { break }
        matchLen += 1
    }
    var flips = 0, firstDivFrame = -1, firstDivCb = -1
    for t in 0 ..< common {
        for c in 0 ..< 32 where fp32.codes[t][c] != bf16.codes[t][c] {
            flips += 1
            if firstDivFrame < 0 { firstDivFrame = t; firstDivCb = c }
        }
    }
    let totalCodes = common * 32
    let pctAgree = 100.0 * Double(totalCodes - flips) / Double(max(totalCodes, 1))
    print(String(format: "  (2) argmax agreement over %d common frames: prefix-exact %d/%d frames · %d/%d codebook flips (%.3f%% agree)",
                 common, matchLen, common, flips, totalCodes, pctAgree))
    if firstDivFrame >= 0 {
        print("  (3) first divergence: frame \(firstDivFrame), codebook \(firstDivCb) "
              + "(fp32=\(fp32.codes[firstDivFrame][firstDivCb]) bf16=\(bf16.codes[firstDivFrame][firstDivCb]))")
    } else {
        print("  (3) first divergence: NONE over the common length — bit-identical argmax")
    }

    // Audio-level divergence (bf16 vs fp32 decoded), scale-invariant, on the common frames.
    let sFp = decodeToSamples(fp32.codes)
    let n = min(sFp.count, sBf.count)
    if n > 0 {
        let a = MLXArray(Array(sBf[0..<n])), b = MLXArray(Array(sFp[0..<n]))
        let ac = a - a.mean(), bc = b - b.mean()
        let corr = (ac * bc).sum().item(Float.self) / (sqrt((ac * ac).sum() * (bc * bc).sum()).item(Float.self) + 1e-12)
        print(String(format: "  (aux) bf16-vs-fp32 decoded-audio corr over %d common samples: %.5f%@",
                     n, corr, fp32.codes.count == bf16.codes.count ? "" : "  (length differs — corr on prefix only)"))
    }

    // (4) Teacher-forced isolation: feed the fp32 rollout's history to a bf16 model and, at each
    // frame, compare bf16's argmax to fp32's chosen code — recording the bf16 logit gap between
    // its own pick and fp32's pick. Gap < TIE ⇒ a near-tie bf16 merely tipped (benign; the
    // free-run divergence is an autoregressive cascade from such tips, not broken numerics).
    // Gap ≥ TIE ⇒ a decisive bf16 argmax error at identical input. (P6's TIE = 5e-3, but that
    // gated fp32-vs-fp32; bf16 eps ≈ 7.8e-3, so we also report the gap distribution, not a verdict.)
    let TIE: Float = 5e-3
    let decBf = GepardDecoder(backbone: bindBackboneDtype(raw, .bfloat16),
                              audio: bindAudioDtype(raw, levels: levels, .bfloat16),
                              heads: bindHeadsDtype(raw, levels: levels, .bfloat16))
    let rcBf = bindRefCompressorDtype(raw, .bfloat16)
    eval(decBf.backbone, decBf.audio, decBf.heads, rcBf)
    let prefixBf = rcBf(refCodes: refCodes).prefix; eval(prefixBf)
    let Tforce = fp32.codes.count
    func fpFrame(_ t: Int) -> [Int] { fp32.codes[t] }
    var tfMismatch = 0, tfDecisive = 0
    var worstTie: Float = 0, worstDecisive: Float = 0
    func score(_ t: Int, _ logits: [MLXArray]) {
        let gf = fpFrame(t)
        for c in 0 ..< 32 {
            let v = logits[c].asType(.float32).reshaped(-1).asArray(Float.self)
            let mine = v.firstIndex(of: v.max()!)!
            if mine != gf[c] {
                tfMismatch += 1
                let gap = v[mine] - v[gf[c]]           // bf16's pick minus fp32's pick (≥ 0)
                if gap >= TIE { tfDecisive += 1; worstDecisive = max(worstDecisive, gap) }
                else { worstTie = max(worstTie, gap) }
            }
        }
    }
    var (hb, cb) = decBf.prefill(prefix: prefixBf, condIds: condIds)
    score(0, decBf.heads(hb).logits)
    for t in 1 ..< Tforce {
        hb = decBf.decodeStep(frameEmbed: decBf.audio.frameEmbed(codes: fpFrame(t - 1)), cache: cb)
        score(t, decBf.heads(hb).logits)
    }
    let tfTotal = Tforce * 32
    print(String(format: "  (4) teacher-forced (bf16 on fp32 history, %d frames): %d/%d flips · %d decisive (gap≥%.0e, worst %.2e) · near-tie worst gap %.2e",
                 Tforce, tfMismatch, tfTotal, tfDecisive, TIE, worstDecisive, worstTie))
    if tfDecisive == 0 {
        print("      → bf16 introduces NO decisive argmax errors at identical input; free-run divergence is a benign AR cascade off near-tie tips.")
    } else {
        print("      → bf16 flips \(tfDecisive) codebook argmax(es) DECISIVELY vs fp32 at identical input (real precision loss, not just ties).")
    }

    print("")
    if audibleBf {
        print("AB-DTYPE PASS — bf16 runtime path yields valid speech; divergence report above.")
        exit(0)
    } else {
        print("AB-DTYPE FAIL — bf16 output not audible-valid")
        exit(1)
    }
}

/// Minimal 16-bit PCM WAV → [Float] decoder (the wrapper emits canonical 16-bit WAV).
func decodeWavSamples(_ data: Data) -> [Float] {
    // Find the "data" chunk.
    let bytes = [UInt8](data)
    guard bytes.count > 44 else { return [] }
    var i = 12
    var dataOffset = 44, dataSize = bytes.count - 44
    while i + 8 <= bytes.count {
        let id = String(bytes: bytes[i..<i+4], encoding: .ascii) ?? ""
        let sz = Int(bytes[i+4]) | Int(bytes[i+5]) << 8 | Int(bytes[i+6]) << 16 | Int(bytes[i+7]) << 24
        if id == "data" { dataOffset = i + 8; dataSize = min(sz, bytes.count - dataOffset); break }
        i += 8 + sz + (sz & 1)
    }
    let n = dataSize / 2
    var out = [Float](repeating: 0, count: n)
    data.withUnsafeBytes { raw in
        let p = raw.baseAddress!.advanced(by: dataOffset).assumingMemoryBound(to: Int16.self)
        for k in 0..<n { out[k] = Float(Int16(littleEndian: p[k])) / 32767.0 }
    }
    return out
}
