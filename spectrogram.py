import numpy as np
import matplotlib.pyplot as plt
import wave

def wav_to_waveform_png(wav_file, output_png):
    # WAV 파일 열기
    with wave.open(wav_file, 'rb') as wf:
        # WAV 파일에서 데이터 가져오기
        num_frames = wf.getnframes()
        framerate = wf.getframerate()
        duration = num_frames / framerate
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        # 음성 데이터 읽기
        frames = wf.readframes(-1)

        # numpy 배열로 변환
        frames_np = np.frombuffer(frames, dtype=np.int16)

        # 시간 배열 생성
        time = np.linspace(0, duration, num_frames)

        # 이미지 크기 설정
        plt.figure(figsize=(12, 6))

        # 다중 채널일 경우 처리
        if channels == 2:
            plt.plot(time, frames_np[::channels], 'steelblue')  # left channel
            plt.plot(time, frames_np[1::channels], 'steelblue')  # right channel
        else:
            plt.plot(time, frames_np, 'steelblue')

        # 축 라벨링 및 제목 설정
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')

        # 이미지 저장
        plt.savefig(output_png)

        # 이미지 출력
        plt.show()

# WAV 파일 경로 및 PNG 파일 경로 설정
wav_file = 'data/audio/00004935.wav'
output_png = 'waveform.png'

# 함수 호출
wav_to_waveform_png(wav_file, output_png)
