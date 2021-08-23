from gevent.server import StreamServer
import Fit_IC
import time


def handler(cs, ca):
    print("사진전송시작")
    file = open("img.jpg", "wb")
    start = time.time()

    msg = cs.recv(1024)
    data = msg

    while msg:
        msg = cs.recv(1024)
        data += msg

    file.write(data)
    file.close()
    pause = time.time()
    save_time = pause - start
    print("저장되는데 걸린시간 : " + str(save_time))
    Cnn = Fit_IC.ImgC("img.jpg")
    result = Cnn.fit()
    stop = time.time()
    run_time = stop - start
    exe_time = stop - pause

    print("머신러닝 실행되는데 걸린시간 : " + str(exe_time))
    print("총 실행되는데 걸린시간 : " + str(run_time))
    file = open("result.txt", "w", encoding="utf-8")
    file.write(result)
    file.close()
    print("계산끝 : 나온 결과는 " + result)

    cs.close()


server = StreamServer(('203.255.39.116', 7321), handler)
print("서버시작")
server.serve_forever()
print("서버시작")