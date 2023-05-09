# Getting Started

required
python >= 3.8

project root 
```bash
source venv/bin/activate
```

종속성 설치
```bash
pip install -r requirements.txt
```

환경 변수 추가 후 파일 실행
```
python3 llm.py
```

배포
```
pm2 start app.py --interpreter python3
```