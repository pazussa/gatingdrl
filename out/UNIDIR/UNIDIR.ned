package inet.showcases.tsn.trafficshaping.Pruebas_tesis.Red_UNIDIR;

import inet.node.ethernet.EthernetLink;
import inet.node.contract.IEthernetNetworkNode;
import inet.node.tsn.TsnDevice;
import inet.node.tsn.TsnSwitch;
import inet.networks.base.TsnNetworkBase;

network unidir extends TsnNetworkBase
{
    parameters:
        *.eth[*].bitrate = default(1000Mbps);
    submodules:
        c1: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,200");
        }
        c2: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,280");
        }
        s1: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,250");
        }
        srv1: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=700,250");
        }

    connections:
        s1.ethg++ <--> EthernetLink <--> srv1.ethg++;
        c1.ethg++ <--> EthernetLink <--> s1.ethg++;
        c2.ethg++ <--> EthernetLink <--> s1.ethg++;
}
