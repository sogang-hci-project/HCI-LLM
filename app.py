from flask_restx import Api, Resource
from flask import Flask, request
from llm import chain, print_result

app = Flask(__name__)
api = Api(app)  # Flask 객체에 Api 객체 등록
app.config["DEBUG"] = True


@app.route("/", methods=["GET"])
def index():
    return "Hello, this is SOGANG HCI project"


@api.route("/hello")
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello": "world!"}

    def post(self):
        question = request.json.get("question")
        result = chain(question)
        res = print_result(result, question)
        return {"contents": res}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
