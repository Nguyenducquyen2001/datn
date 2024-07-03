import serial
import csv
import struct
import time
import sys
import signal


BUFF_LEN = 512
TIME     = 2

ser = serial.Serial('COM5', 921600) 
    
ser.read_until(b'!')
while True:

    with open('./gui/data2.csv', 'a', newline='') as csvfile2,\
        open('./gui/data3.csv', 'a', newline='') as csvfile3:
        
        writer2 = csv.writer(csvfile2)
        writer3 = csv.writer(csvfile3)
    
        def read_mpu6050():   
            data1 = ser.read_until(b'd')           
            data = ser.readline().decode('utf-8').strip()
            data2 = ser.read_until(b'e')
            parts = data.split(',')
            writer2.writerow(parts)
            print("đã ghi data MPU6050")

        def read_inmp441():   
                
            data1 = ser.read_until(b'd')     
            data = ser.read(BUFF_LEN * 2)  # đọc 20 byte
            data2 = ser.read_until(b'e')  
            values = struct.unpack('<'+'h' * (BUFF_LEN), data)  # giải nén dữ liệu
            values_row = [value for value in values]

            writer3.writerow(values_row) 
            print("đã ghi data âm thanh")
        
        start_time = time.time()
        while ((time.time() - start_time) < TIME):
            if ser.in_waiting:
                match ser.read(1):
                    case b'm':
                        read_mpu6050()

                    case b'c':
                        read_inmp441()
        # csvfile2.close()
        # csvfile3.close()
                        
        # ser.close()
        # sys.exit(0)


    

         
