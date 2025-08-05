package inet.showcases.tsn.trafficshaping.Pruebas_tesis.Red_UNIDIR8;

import inet.node.ethernet.EthernetLink;
import inet.node.contract.IEthernetNetworkNode;
import inet.node.tsn.TsnDevice;
import inet.node.tsn.TsnSwitch;
import inet.networks.base.TsnNetworkBase;

infrastructure unidir8 extends TsnNetworkBase
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
        c3: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,360");
        }
        c4: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,440");
        }
        c5: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,520");
        }
        c6: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,600");
        }
        c7: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,680");
        }
        c8: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=300,760");
        }
        s1: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,250");
        }
        s2: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,330");
        }
        s3: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,410");
        }
        s4: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,490");
        }
        s5: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,570");
        }
        s6: <default("TsnSwitch")> like IEthernetNetworkNode {
            @display("p=500,650");
        }
        srv1: <default("TsnDevice")> like IEthernetNetworkNode {
            @display("p=700,250");
        }

    connections:
        s1.ethg++ <--> EthernetLink <--> s5.ethg++;
        s2.ethg++ <--> EthernetLink <--> s5.ethg++;
        s3.ethg++ <--> EthernetLink <--> s5.ethg++;
        s4.ethg++ <--> EthernetLink <--> s5.ethg++;
        s5.ethg++ <--> EthernetLink <--> s6.ethg++;
        s6.ethg++ <--> EthernetLink <--> srv1.ethg++;
        c1.ethg++ <--> EthernetLink <--> s1.ethg++;
        c2.ethg++ <--> EthernetLink <--> s1.ethg++;
        c3.ethg++ <--> EthernetLink <--> s2.ethg++;
        c4.ethg++ <--> EthernetLink <--> s2.ethg++;
        c5.ethg++ <--> EthernetLink <--> s3.ethg++;
        c6.ethg++ <--> EthernetLink <--> s3.ethg++;
        c7.ethg++ <--> EthernetLink <--> s4.ethg++;
        c8.ethg++ <--> EthernetLink <--> s4.ethg++;
}
