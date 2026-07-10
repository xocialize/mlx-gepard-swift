// AudioSupport.swift — canonical Audio codec + reference-audio resampling for the wrapper.
//
// decodeToMono / encodeWAV16 follow the fleet pattern (MLXIndexTTS2TTS / MLXVoxCPM2TTS);
// SincResampler is the fleet's inline windowed-sinc kernel (matches librosa/resample_poly to
// float32 rounding). Gepard has no mel/STFT, so the wrapper's ONLY DSP need is resampling an
// arbitrary reference clip to the codec's 22.05 kHz — hence no mlx-audio-dsp dependency.

import AVFoundation
import Foundation
import MLXToolKit

enum AudioSupport {

    enum AudioError: Error, CustomStringConvertible {
        case unreadableReferenceAudio(String)
        var description: String {
            switch self {
            case .unreadableReferenceAudio(let why): return "unreadable reference audio: \(why)"
            }
        }
    }

    /// Decodes a canonical `Audio` (.wav) artifact to mono float samples + sample rate.
    /// Multi-channel input is mixed down by averaging (the oracle pipeline's librosa mono mix).
    static func decodeToMono(_ audio: Audio) throws -> (samples: [Float], sampleRate: Int) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("gepard-ref-\(UUID().uuidString).wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: tmp)
        } catch {
            throw AudioError.unreadableReferenceAudio(error.localizedDescription)
        }
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw AudioError.unreadableReferenceAudio("empty audio")
        }
        try file.read(into: buffer)

        let channels = Int(format.channelCount)
        let count = Int(buffer.frameLength)
        guard channels > 0, count > 0, let channelData = buffer.floatChannelData else {
            throw AudioError.unreadableReferenceAudio("no decodable samples")
        }
        var mono = [Float](repeating: 0, count: count)
        for channel in 0 ..< channels {
            let p = channelData[channel]
            for i in 0 ..< count { mono[i] += p[i] }
        }
        if channels > 1 {
            let inv = 1 / Float(channels)
            for i in 0 ..< count { mono[i] *= inv }
        }
        return (mono, Int(format.sampleRate))
    }

    /// Encodes mono float samples as a 16-bit PCM WAV (broadly playable) in memory.
    static func encodeWAV16(samples: [Float], sampleRate: Int) -> Data {
        let channels = 1
        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = samples.count * blockAlign

        var data = Data(capacity: 44 + dataSize)
        func ascii(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        ascii("RIFF"); u32(UInt32(36 + dataSize)); ascii("WAVE")
        ascii("fmt "); u32(16); u16(1) // PCM
        u16(UInt16(channels)); u32(UInt32(sampleRate)); u32(UInt32(byteRate))
        u16(UInt16(blockAlign)); u16(UInt16(bitsPerSample))
        ascii("data"); u32(UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            var le = Int16(clamped * 32767).littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Peak-normalized RMS in dBFS of mono samples — the app harness's non-silence probe
    /// (a silent stem reads −∞; don't trust ears). Returns -.infinity for empty/silent input.
    static func dBFS(_ samples: [Float]) -> Double {
        guard !samples.isEmpty else { return -.infinity }
        let sumSq = samples.reduce(0.0) { $0 + Double($1) * Double($1) }
        let rms = (sumSq / Double(samples.count)).squareRoot()
        return rms > 0 ? 20.0 * Foundation.log10(rms) : -.infinity
    }
}

/// High-quality mono resampler using a Hann-windowed sinc kernel (fleet-standard for
/// voice-cloning reference audio — aliasing directly degrades cloned timbre).
enum SincResampler {

    /// Resample mono audio from `fromRate` to `toRate`. Default 32 taps/side ≈ 96 dB
    /// stop-band rejection — transparent for speech.
    static func resample(
        audio: [Float], from fromRate: Int, to toRate: Int, tapsPerSide: Int = 32
    ) -> [Float] {
        if fromRate == toRate { return audio }
        if audio.isEmpty { return [] }

        let ratio = Double(toRate) / Double(fromRate)
        let outputLen = Int((Double(audio.count) * ratio).rounded())
        // Downsampling narrows the sinc cutoff to the new Nyquist (anti-aliasing).
        let cutoff: Double = ratio < 1.0 ? ratio : 1.0

        var output = [Float](repeating: 0, count: outputLen)
        let srcLen = audio.count

        for i in 0 ..< outputLen {
            let srcPos = Double(i) / ratio
            let centerIdx = Int(srcPos)
            let frac = srcPos - Double(centerIdx)

            var sum: Double = 0
            var normalization: Double = 0
            for k in -tapsPerSide ... tapsPerSide {
                let idx = centerIdx + k
                if idx < 0 || idx >= srcLen { continue }
                let x = Double(k) - frac
                let sincVal = sinc(cutoff * x)
                let windowPos = Double(k) / Double(tapsPerSide + 1)
                let hann = 0.5 * (1.0 + cos(.pi * windowPos))
                let weight = sincVal * hann * cutoff
                sum += Double(audio[idx]) * weight
                normalization += weight
            }
            // Normalize to preserve DC gain (eliminates subtle level drift).
            output[i] = normalization > 1e-10 ? Float(sum / normalization) : 0
        }
        return output
    }

    @inline(__always)
    private static func sinc(_ x: Double) -> Double {
        if abs(x) < 1e-12 { return 1.0 }
        let px = .pi * x
        return sin(px) / px
    }
}
