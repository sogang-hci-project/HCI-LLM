# Getting Started

required
python >= 3.8

가상환경 생성
```bash
python3 -m venv venv
```

project root 
```bash
source venv/bin/activate
```

종속성 설치
```bash
pip3 install -r requirements.txt
```

환경 변수 추가 후 파일 실행
```
OPEN_AI_API_KEY=''
```
```
python3 llm.py
```

로컬 실행
```
python3 app.py
```

배포
```
pm2 start app.py --interpreter python3
```