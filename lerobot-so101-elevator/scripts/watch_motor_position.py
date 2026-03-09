import serial
import time
import argparse
import sys

def calculate_checksum(packet):
    return (~sum(packet[2:-1])) & 0xFF

def send_packet(ser, servo_id, instruction, parameters):
    length = len(parameters) + 2
    packet = [0xFF, 0xFF, servo_id, length, instruction] + parameters + [0]
    packet[-1] = calculate_checksum(packet)
    ser.write(bytearray(packet))

def read_packet(ser, servo_id, address, length):
    send_packet(ser, servo_id, 0x02, [address, length])
    time.sleep(0.02)
    if ser.in_waiting:
        response = ser.read(ser.in_waiting)
        if len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF:
            # Response: 0xFF 0xFF ID LEN ERR PARAM... CHK
            return list(response[5:-1])
    return None

def main():
    parser = argparse.ArgumentParser(description='Watch Feetech STS3215 motor position in real-time.')
    parser.add_argument('--port', type=str, required=True, help='Serial port (e.g., /dev/ttyACM0)')
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='Motor IDs to monitor (default: 1 2 3 4 5 6)')
    
    args = parser.parse_args()
    
    try:
        ser = serial.Serial(args.port, 1000000, timeout=0.1)
        print(f"Monitoring Motor IDs {args.ids} on {args.port}. Press Ctrl+C to stop.")
        
        while True:
            output_str = "\r"
            for motor_id in args.ids:
                # Address 56 is Current Position (2 bytes)
                pos_data = read_packet(ser, motor_id, 56, 2)
                if pos_data and len(pos_data) >= 2:
                    position = pos_data[0] | (pos_data[1] << 8)
                    
                    # Convert to signed if using multi-turn mode and value is negative
                    if position > 32767:
                        position -= 65536
                        
                    output_str += f"ID {motor_id}: {position:5d} | "
                else:
                    output_str += f"ID {motor_id}: Error | "
                    
            sys.stdout.write(output_str)
            sys.stdout.flush()
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
