import serial
import time
import argparse

def calculate_checksum(packet):
    # Checksum = ~(ID + Length + Instruction + Parameter1 + ... + ParameterN) & 0xFF
    return (~sum(packet[2:-1])) & 0xFF

def send_packet(ser, servo_id, instruction, parameters):
    length = len(parameters) + 2
    packet = [0xFF, 0xFF, servo_id, length, instruction] + parameters + [0]
    packet[-1] = calculate_checksum(packet)
    
    print(f"Sending packet: {[hex(b) for b in packet]}")
    ser.write(bytearray(packet))
    time.sleep(0.05)
    
    # Read response (optional)
    if ser.in_waiting:
        response = ser.read(ser.in_waiting)
        print(f"Response: {[hex(b) for b in response]}")

def main():
    parser = argparse.ArgumentParser(description='Change Feetech STS3215 motor mode from Extended Position (4) to Position (0).')
    parser.add_argument('--port', type=str, required=True, help='Serial port (e.g., /dev/ttyACM0)')
    parser.add_argument('--id', type=int, default=5, help='Motor ID (default: 5)')
    
    args = parser.parse_args()
    
    try:
        ser = serial.Serial(args.port, 1000000, timeout=1)
        print(f"Opened port {args.port} at 1M baud.")
        
        # 1. Unlock EPROM (Address 55 = 0)
        print(f"Step 1: Unlocking EPROM for ID {args.id}...")
        send_packet(ser, args.id, 0x03, [55, 0])
        
        # 2. Set Operating Mode to Position Mode (Address 33 = 0)
        print(f"Step 2: Setting Operating Mode to 0 (Position Mode)...")
        send_packet(ser, args.id, 0x03, [33, 0])
        
        # 3. Lock EPROM (Address 55 = 1)
        print(f"Step 3: Locking EPROM...")
        send_packet(ser, args.id, 0x03, [55, 1])
        
        print("\nSuccess! Please REBOOT (power cycle) your motor for changes to take effect.")
        print("After rebooting, the motor should stay within 0-4095 range and NOT reset its position to 2048 at boot.")
        
        ser.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
