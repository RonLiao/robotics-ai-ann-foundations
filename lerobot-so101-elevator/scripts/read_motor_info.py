import serial
import time
import argparse

def calculate_checksum(packet):
    return (~sum(packet[2:-1])) & 0xFF

def send_packet(ser, servo_id, instruction, parameters):
    length = len(parameters) + 2
    packet = [0xFF, 0xFF, servo_id, length, instruction] + parameters + [0]
    packet[-1] = calculate_checksum(packet)
    ser.write(bytearray(packet))
    time.sleep(0.1)

def read_packet(ser, servo_id, address, length):
    send_packet(ser, servo_id, 0x02, [address, length])
    time.sleep(0.1)
    if ser.in_waiting:
        response = ser.read(ser.in_waiting)
        if len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF:
            # Response: 0xFF 0xFF ID LEN ERR PARAM... CHK
            # Parameters start at index 5
            return list(response[5:-1])
    return None

def main():
    parser = argparse.ArgumentParser(description='Read Feetech STS3215 motor registers.')
    parser.add_argument('--port', type=str, required=True, help='Serial port')
    parser.add_argument('--id', type=int, default=5, help='Motor ID')
    
    args = parser.parse_args()
    
    try:
        ser = serial.Serial(args.port, 1000000, timeout=1)
        
        # Read Operating Mode (Address 33)
        mode_data = read_packet(ser, args.id, 33, 1)
        mode = mode_data[0] if mode_data else "Unknown"
        
        # Read Homing Offset (Address 20, 2 bytes)
        homing_data = read_packet(ser, args.id, 20, 2)
        homing = (homing_data[0] | (homing_data[1] << 8)) if homing_data else "Unknown"
        # Convert to signed int16 if necessary
        if isinstance(homing, int) and homing > 32767:
            homing -= 65536

        print("\n" + "="*40)
        print(f"Motor ID: {args.id}")
        print(f"Current Operating Mode (Addr 33): {mode}")
        print(f"Homing Offset (Addr 20): {homing}")
        print("="*40 + "\n")
        
        if mode == 0:
            print("Mode 0: Position Mode (0-4095 absolute)")
        elif mode == 4:
            print("Mode 4: Extended Position Mode (Multi-turn)")
            
        ser.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
