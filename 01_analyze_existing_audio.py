#音声波形、有声・無声の区別、有声確率、基本周波数、スペクトログラムを表示
#このスクリプトでは録音済みの音声を処理している。

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 16,          # 基本の文字サイズ
    "axes.titlesize": 20,     # 各グラフのタイトル
    "axes.labelsize": 18,     # x軸・y軸ラベル
    "xtick.labelsize": 15,    # x軸目盛
    "ytick.labelsize": 15,    # y軸目盛
    "legend.fontsize": 15,    # 凡例
    "figure.titlesize": 22,   # 図全体のタイトル
})
# ==================== 設定 ====================
AUDIO_PATH = "recorded_audio.wav"
TARGET_SAMPLE_RATE = 16000

# F0推定用
F0_FRAME_LENGTH = 1024
F0_HOP_LENGTH = 160       # 10 ms
F0_MIN = 70.0
F0_MAX = 500.0

# スペクトログラム用
SPEC_WIN_LENGTH = 400     # 25 ms
SPEC_HOP_LENGTH = 80      # 5 ms
SPEC_N_FFT = 512

FIG_PATH = "voice_analysis.png"
# ==============================================


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    audio, sample_rate = librosa.load(
        path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("音声データが空です。")

    return audio, sample_rate


def analyze_f0(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sample_rate,
        frame_length=F0_FRAME_LENGTH,
        hop_length=F0_HOP_LENGTH,
    )

    frame_times = librosa.times_like(
        f0,
        sr=sample_rate,
        hop_length=F0_HOP_LENGTH,
    )

    return frame_times, f0, voiced_flag, voiced_prob


def calculate_spectrogram(
    audio: np.ndarray,
) -> np.ndarray:
    stft = librosa.stft(
        audio,
        n_fft=SPEC_N_FFT,
        win_length=SPEC_WIN_LENGTH,
        hop_length=SPEC_HOP_LENGTH,
        window="hann",
    )

    return librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max,
    )


def plot_analysis(
    audio: np.ndarray,
    sample_rate: int,
    frame_times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_prob: np.ndarray,
    spectrogram_db: np.ndarray,
) -> None:
    duration = len(audio) / sample_rate
    waveform_times = np.arange(len(audio)) / sample_rate

    # 右側にカラーバー専用列を設けることで、
    # 4つの主グラフの横幅をそろえる
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=[1.0, 0.025],
        height_ratios=[1, 1, 1, 1],
        hspace=0.35,
        wspace=0.08,
    )

    ax_wave = fig.add_subplot(grid[0, 0])
    ax_voice = fig.add_subplot(grid[1, 0], sharex=ax_wave)
    ax_f0 = fig.add_subplot(grid[2, 0], sharex=ax_wave)
    ax_spec = fig.add_subplot(grid[3, 0], sharex=ax_wave)
    cax = fig.add_subplot(grid[3, 1])

    # 波形
    ax_wave.plot(waveform_times, audio, linewidth=0.7)
    ax_wave.set_ylim(-0.25, 0.25)
    ax_wave.set_yticks([-0.25, 0, 0.25])
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform")
    ax_wave.set_xlim(0, duration)
    ax_wave.grid(True, alpha=0.3)

    # 有声・無声
    ax_voice.step(
        frame_times,
        voiced_flag.astype(int),
        where="mid",
        label="Voiced flag",
    )
    ax_voice.plot(
        frame_times,
        voiced_prob,
        linewidth=1.0,
        alpha=0.8,
        color="red",
        label="Voiced probability",
    )
    ax_voice.set_ylim(-0.1, 1.1)
    ax_voice.set_ylabel("Voice")
    ax_voice.set_title("Voiced / Unvoiced")
    ax_voice.grid(True, alpha=0.3)
    legend = ax_voice.legend(
        loc="upper left",
        bbox_to_anchor=(0.93, 1.0),
        borderaxespad=0,
        frameon=True,
        edgecolor="black",
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    

    # 通常の目盛ラベルを消し、グラフ内側に配置
    ax_voice.set_yticks(
        [0, 1],
        labels=["Unvoiced", "Voiced"],
        )
    
    ax_voice.set_ylabel("Voice")
    ax_voice.yaxis.set_label_coords(-0.035, 0.5)

    # F0
    ax_f0.plot(frame_times, f0, linewidth=1.5)
    ax_f0.set_ylim(F0_MIN, F0_MAX)
    ax_f0.set_ylim(F0_MIN, F0_MAX)
    ax_f0.set_yticks([F0_MIN, 200, 300, 400, F0_MAX])
    ax_f0.set_ylabel("F0 [Hz]")
    ax_f0.set_title("Fundamental Frequency")
    ax_f0.grid(True, alpha=0.3)

    # スペクトログラム
    image = librosa.display.specshow(
        spectrogram_db,
        sr=sample_rate,
        n_fft=SPEC_N_FFT,
        hop_length=SPEC_HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax_spec,
    )
    ax_spec.set_ylim(0, sample_rate / 2)
    ax_spec.set_xlim(0, duration)
    ax_spec.set_title("Spectrogram")
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylabel("Frequency [Hz]")

    fig.colorbar(
        image,
        cax=cax,
        format="%+2.0f dB",
        label="Amplitude [dB]",
    )

    # 上3段ではx軸の数値を非表示
    plt.setp(ax_wave.get_xticklabels(), visible=False)
    plt.setp(ax_voice.get_xticklabels(), visible=False)
    plt.setp(ax_f0.get_xticklabels(), visible=False)

    fig.savefig(
        FIG_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"解析画像を保存しました: {FIG_PATH}")
    plt.show()


def main() -> None:
    try:
        audio, sample_rate = load_audio(AUDIO_PATH)

        print(f"読み込みファイル: {AUDIO_PATH}")
        print(f"サンプリング周波数: {sample_rate} Hz")
        print(f"音声長: {len(audio) / sample_rate:.2f} 秒")

        frame_times, f0, voiced_flag, voiced_prob = analyze_f0(
            audio,
            sample_rate,
        )
        spectrogram_db = calculate_spectrogram(audio)

        plot_analysis(
            audio,
            sample_rate,
            frame_times,
            f0,
            voiced_flag,
            voiced_prob,
            spectrogram_db,
        )

    except FileNotFoundError as error:
        print(error)
    except Exception as error:
        print(f"処理中にエラーが発生しました: {error}")


if __name__ == "__main__":
    main()
