import speech_recognition as sr
from pydub import AudioSegment

# 加载 WAV 文件
audio = AudioSegment.from_wav(r"C:\Users\陈宇飞\Desktop\大会.wav")

# 将音频分割为 30 秒的小片段
segment_length_ms = 60000  # 30 秒
segments = [audio[i:i + segment_length_ms] for i in range(0, len(audio), segment_length_ms)]

# 创建识别器
recognizer = sr.Recognizer()

for i, segment in enumerate(segments):
    segment.export(f"segment_{i}.wav", format="wav")

    # 使用 Google Web Speech API 进行语音识别
    with sr.AudioFile(f"segment_{i}.wav") as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language='zh-CN')
        print(f"段 {i + 1} 识别结果: " + text)
    except sr.UnknownValueError:
        print(f"段 {i + 1} 无法识别")
    except sr.RequestError as e:
        print(f"段 {i + 1} 请求错误: {e}")

