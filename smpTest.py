import smp
import socket
import sys
import M2Crypto

# Check command line args
if len(sys.argv) != 2:
    print("Usage: %s [IP/listen]" % sys.argv[0])
    sys.exit(1)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

if sys.argv[1] == 'listen':
    # Listen for incoming connections
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', 5000))
    sock.listen(1)
    print("Listening for client")
    client = sock.accept()[0]

    # Prompt the user for a shared secret to use in SMP
    secret = input("Enter shared secret: ")

    # Create an SMP object with the calculated secret
    smp = smp.SMP(secret)

    # Do the SMP protocol
    buffer = client.recv(4096)
    buffer = smp.step2(buffer)
    client.send(buffer)

    buffer = client.recv(4096)
    buffer = smp.step4(buffer)
    client.send(buffer)
else:
    # Connect to the server
    sock.connect((sys.argv[1], 5000))

    # Prompt the user for a shared secret to use in SMP
    secret = input("Enter shared secret: ")

    # Create an SMP object with the calculated secret
    smp = smp.SMP(secret)

    # Do the SMP protocol
    buffer = smp.step1()
    sock.send(buffer)

    buffer = sock.recv(4096)
    buffer = smp.step3(buffer)
    sock.send(buffer)

    buffer = sock.recv(4096)
    smp.step5(buffer)

# Check if the secrets match
if smp.match:
    print("Secrets match")
else:
    print("Secrets do not match")
