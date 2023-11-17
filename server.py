import numpy as np
import cv2
import socket
from ultralytics import YOLO

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the port
server_address = ('192.168.0.101', 7777) 
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)
def applyImage(data):
	decoded = np.frombuffer(data, dtype=np.uint8)
	decoded = decoded.reshape((480, 640,3))
	return decoded

def readnbyte(sock, n):
    buff = bytearray(n)
    pos = 0
    while pos < n:
        cr = sock.recv_into(memoryview(buff)[pos:])
        if cr == 0:
            raise EOFError
        pos += cr
    return buff

names = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt","yolov8x.pt"]

model = YOLO(names[0])
model.conf = 0.75    
while True:
    # Wait for a connection
    print( 'waiting for a connection')
    connection, client_address = sock.accept()
    connection.setblocking(True)
    try:
        print( 'connection from', client_address)
        # Receive the data in once 
        while True:
            l = 0
            img = bytearray(0)
            l_bytes = connection.recv(4)
            size = int.from_bytes(l_bytes, "little")
            print('size = ', size)
            img = readnbyte(connection, size)
            cv2.imshow('IMG',model.predict(applyImage(img))[0].plot())
            cv2.waitKey(1)
    finally:
        # Clean up the connection
        connection.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
connection.close()