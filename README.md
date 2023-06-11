# nexon_prototype
 prototype_for_recommend

python == 3.8.12

in Mac

pip install virtualenv

brew install pyenv   # Mac에서 pyenv 설치
pyenv install 3.8.12  # Python 3.8.12 설치

pyenv local 3.8.12  # 현재 디렉토리에 Python 버전을 3.8.12로 설정
virtualenv venv  # venv라는 이름의 가상 환경 생성
source venv/bin/activate  # 가상 환경 활성화

pip install -r ./requirment/requirements.txt
