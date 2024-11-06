import subprocess
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

def check_gpu_memory_usage():
    # nvidia-smi 명령어 실행
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
    output = result.stdout
    
    # 메모리 사용량 파싱
    memory_info = re.search(r"(\d+)MiB / (\d+)MiB", output)
    if memory_info:
        memory_used = int(memory_info.group(1))
        memory_total = int(memory_info.group(2))
        usage_percentage = (memory_used / memory_total) * 100
        return f"GPU Memory Usage: {memory_used} MiB / {memory_total} MiB ({usage_percentage:.2f}%)"
    return "No GPU information found."

@app.route("/slack/gpu_usage", methods=["POST"])
def gpu_usage():
    # GPU 정보 확인
    gpu_info = check_gpu_memory_usage()
    
    # Slack에 JSON 형식으로 응답
    return jsonify({
        "response_type": "in_channel",
        "text": gpu_info
    })

if __name__ == "__main__":
    app.run(port=5000)

# 슬랙 앱 생성
# request url에 터널링 서버 주소 입력
# 슬랙에서 만든 앱 설치 및 초대 

# ngrok에서 터널링 생성 -> url

