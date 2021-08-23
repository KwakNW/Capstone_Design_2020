from gevent.server import StreamServer
import time


def handler(cs, ca):
    print("결과전송시작")
    file = open("result.txt", "r", encoding="utf-8")
    result = file.read()
    file.close()
    cs.sendall(result.encode())
    cs.close()
    print("보내기끝")


server = StreamServer(('203.255.39.116', 7320), handler)
print("서버시작")
server.serve_forever()

