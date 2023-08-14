import os
import time
from argparse import ArgumentParser

# from mininet.cli import CLI
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.topo import Topo
from mininet.util import dumpNodeConnections


class StarTopo(Topo):

    def build(self, n=1, bw=None, delay=None):

        # create hosts
        server = self.addHost(f'server')
        clients = []
        for i in range(n):
            client = self.addHost(f'client{i}')
            clients.append(client)

        # create switches
        switch = self.addSwitch('s0')

        # create links
        self.addLink(server, switch)
        link_args = {}
        if bw is not None:
            link_args['bw'] = bw
        if delay is not None:
            link_args['delay'] = f"{delay}ms"
        for client in clients:
            self.addLink(client, switch, cls=TCLink, **link_args)


def test(n=3, bw=None, delay=None):

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mininet_log_{timestamp}.log"
    timestamp = time.strftime("%Y/%m/%d %H:%M:%S")
    bw_limit_str = f"none" if bw is None else f"{bw}Mbps"
    delay_limit_str = f"none" if delay is None else f"{delay}ms"
    with open(filename, "w") as f:
        f.write(f"Mininet experiment started at {timestamp}\n")
        f.write(f"Number of clients {n}\n")
        f.write(f"Bandwidth limit: {bw_limit_str}\n")
        f.write(f"Delay limit: {delay_limit_str}\n")
        f.write(f"Iperf & ping:\n")

    topo = StarTopo(n, bw, delay)
    net = Mininet(topo)
    net.start()

    print("Dumping host connections")
    dumpNodeConnections(net.hosts)
    print("Testing network connectivity")
    net.pingAll()

    server = net.get('server')
    clients = [net.get(f'client{i}') for i in range(n)]

    server.cmd(f"iperf -s >> {filename} &")
    clients[0].cmd(f"iperf -c {server.IP()} > /dev/null")
    clients[0].cmd(f"ping -c 5 {server.IP()} >> {filename}")

    server.sendCmd("./build/server > server.out 2>&1")
    for i in range(n):
        clients[i].sendCmd(
            f"./build/client {server.IP()} {i} > client{i}.out 2>&1")

    server.waitOutput()
    for i in range(n):
        clients[i].waitOutput()

    # server.cmd("./build/server > server.out 2>&1 &")

    # for i in range(n):
    #     clients[i].cmd(
    #         f"./build/client {server.IP()} {i} > client{i}.out 2>&1 &")

    # CLI(net)

    net.stop()

    with open(filename, "a") as f:
        timestamp = time.strftime("%Y/%m/%d %H:%M:%S")
        f.write(f"Mininet experiment ended at {timestamp}\n")


if __name__ == '__main__':

    parser = ArgumentParser(description="Mininet simulation")
    parser.add_argument("-n", "--num-clients", type=int,
                        help="number of clients")
    parser.add_argument("-b", "--bandwidth", type=float,
                        help="link bandwidth (Mbps)")
    parser.add_argument("-d", "--delay", type=str, help="link delay (ms)")
    args = parser.parse_args()

    n = args.num_clients if args.num_clients else 3
    bw = args.bandwidth if args.bandwidth else None
    delay = args.delay if args.delay else None

    test(n, bw, delay)

# sudo mn -c
# nohup sudo python src/mininet_script.py -n 3 -b 1000 -d 10 &
