module fuse_top (clk,
    done,
    rst,
    task_ready,
    task_valid,
    task_data,
    task_hash,
    task_id,
    task_latency,
    task_template);
 input clk;
 output done;
 input rst;
 output task_ready;
 input task_valid;
 input [127:0] task_data;
 input [31:0] task_hash;
 input [7:0] task_id;
 input [15:0] task_latency;
 input [7:0] task_template;

 wire _000_;
 wire _001_;
 wire _002_;
 wire _003_;
 wire _004_;
 wire _005_;
 wire _006_;
 wire _007_;
 wire _008_;
 wire _009_;
 wire _010_;
 wire _011_;
 wire _012_;
 wire _013_;
 wire _014_;
 wire _015_;
 wire _016_;
 wire _017_;
 wire _018_;
 wire _019_;
 wire _020_;
 wire _021_;
 wire _022_;
 wire _023_;
 wire _024_;
 wire _025_;
 wire _026_;
 wire _027_;
 wire _028_;
 wire _029_;
 wire _030_;
 wire _031_;
 wire _032_;
 wire _033_;
 wire _034_;
 wire _035_;
 wire _036_;
 wire _037_;
 wire _038_;
 wire _039_;
 wire _040_;
 wire _041_;
 wire _042_;
 wire _043_;
 wire _044_;
 wire _045_;
 wire _046_;
 wire _047_;
 wire _048_;
 wire _049_;
 wire _050_;
 wire _051_;
 wire _052_;
 wire _053_;
 wire _054_;
 wire _055_;
 wire _056_;
 wire _057_;
 wire _058_;
 wire _059_;
 wire _060_;
 wire _061_;
 wire _062_;
 wire _063_;
 wire _064_;
 wire _065_;
 wire _066_;
 wire _067_;
 wire _068_;
 wire _069_;
 wire _071_;
 wire _074_;
 wire _075_;
 wire _076_;
 wire _077_;
 wire _078_;
 wire _080_;
 wire _081_;
 wire _083_;
 wire _084_;
 wire _085_;
 wire _086_;
 wire _088_;
 wire _089_;
 wire _090_;
 wire _091_;
 wire _092_;
 wire _093_;
 wire _095_;
 wire _096_;
 wire _097_;
 wire _098_;
 wire _099_;
 wire _100_;
 wire _101_;
 wire _102_;
 wire _103_;
 wire _104_;
 wire _105_;
 wire _106_;
 wire _107_;
 wire _108_;
 wire _109_;
 wire _110_;
 wire _112_;
 wire _113_;
 wire _114_;
 wire _115_;
 wire _116_;
 wire _117_;
 wire _118_;
 wire _119_;
 wire _120_;
 wire _121_;
 wire _122_;
 wire _123_;
 wire _124_;
 wire _125_;
 wire _126_;
 wire _127_;
 wire _128_;
 wire _130_;
 wire _131_;
 wire _132_;
 wire _133_;
 wire _134_;
 wire _135_;
 wire _137_;
 wire _138_;
 wire _139_;
 wire _140_;
 wire _141_;
 wire _142_;
 wire _143_;
 wire _144_;
 wire _145_;
 wire _146_;
 wire _147_;
 wire _148_;
 wire _149_;
 wire _150_;
 wire _151_;
 wire _152_;
 wire _153_;
 wire _154_;
 wire _155_;
 wire _156_;
 wire _157_;
 wire _158_;
 wire _159_;
 wire _160_;
 wire _161_;
 wire _162_;
 wire _163_;
 wire _164_;
 wire _165_;
 wire _166_;
 wire _167_;
 wire _168_;
 wire _169_;
 wire _170_;
 wire _171_;
 wire _172_;
 wire _173_;
 wire _174_;
 wire _175_;
 wire _176_;
 wire _177_;
 wire _178_;
 wire _179_;
 wire _180_;
 wire _181_;
 wire _182_;
 wire _183_;
 wire _184_;
 wire _185_;
 wire _186_;
 wire _187_;
 wire _189_;
 wire _190_;
 wire _191_;
 wire _192_;
 wire _193_;
 wire _194_;
 wire _195_;
 wire _196_;
 wire _197_;
 wire _198_;
 wire _199_;
 wire _200_;
 wire _201_;
 wire _202_;
 wire _203_;
 wire _204_;
 wire _205_;
 wire _206_;
 wire _207_;
 wire _208_;
 wire _209_;
 wire _210_;
 wire _211_;
 wire _212_;
 wire _213_;
 wire _214_;
 wire _215_;
 wire _216_;
 wire _217_;
 wire _218_;
 wire _219_;
 wire _220_;
 wire _221_;
 wire _222_;
 wire _223_;
 wire _224_;
 wire _225_;
 wire _226_;
 wire _227_;
 wire _228_;
 wire _229_;
 wire _230_;
 wire _231_;
 wire _232_;
 wire _233_;
 wire _234_;
 wire _235_;
 wire _236_;
 wire _237_;
 wire \active_template[0] ;
 wire \active_template[1] ;
 wire \active_template[2] ;
 wire \active_template[3] ;
 wire \active_template[4] ;
 wire \active_template[5] ;
 wire \active_template[6] ;
 wire \active_template[7] ;
 wire net27;
 wire \g_rfu[0].u_rfu.issue_valid ;
 wire \g_rfu[0].u_rfu.remaining[0] ;
 wire \g_rfu[0].u_rfu.remaining[10] ;
 wire \g_rfu[0].u_rfu.remaining[11] ;
 wire \g_rfu[0].u_rfu.remaining[12] ;
 wire \g_rfu[0].u_rfu.remaining[13] ;
 wire \g_rfu[0].u_rfu.remaining[14] ;
 wire \g_rfu[0].u_rfu.remaining[15] ;
 wire \g_rfu[0].u_rfu.remaining[1] ;
 wire \g_rfu[0].u_rfu.remaining[2] ;
 wire \g_rfu[0].u_rfu.remaining[3] ;
 wire \g_rfu[0].u_rfu.remaining[4] ;
 wire \g_rfu[0].u_rfu.remaining[5] ;
 wire \g_rfu[0].u_rfu.remaining[6] ;
 wire \g_rfu[0].u_rfu.remaining[7] ;
 wire \g_rfu[0].u_rfu.remaining[8] ;
 wire \g_rfu[0].u_rfu.remaining[9] ;
 wire lrc_ready;
 wire \rfu_busy[0][0] ;
 wire \rfu_ready[0][0] ;
 wire net1;
 wire sched_valid;
 wire net2;
 wire net3;
 wire net4;
 wire net5;
 wire net6;
 wire net7;
 wire net8;
 wire net9;
 wire net10;
 wire net11;
 wire net12;
 wire net13;
 wire net14;
 wire net15;
 wire net16;
 wire net17;
 wire net28;
 wire net18;
 wire net19;
 wire net20;
 wire net21;
 wire net22;
 wire net23;
 wire net24;
 wire net25;
 wire net26;
 wire \u_lrc.busy ;
 wire \u_lrc.counter[0] ;
 wire \u_lrc.counter[1] ;
 wire \u_lrc.counter[2] ;
 wire \u_lrc.counter[3] ;
 wire \u_lrc.counter[4] ;
 wire \u_lrc.counter[5] ;
 wire \u_lrc.counter[6] ;
 wire \u_lrc.counter[7] ;
 wire \u_tars.count[0] ;
 wire \u_tars.count[1] ;
 wire \u_tars.count[2] ;
 wire \u_tars.count[3] ;
 wire net41;
 wire net36;
 wire net37;
 wire net39;
 wire clknet_0_clk;
 wire net35;
 wire net38;
 wire net40;
 wire net42;
 wire clknet_2_0_0_clk;
 wire clknet_2_1_0_clk;
 wire clknet_2_2_0_clk;
 wire clknet_2_3_0_clk;

 sky130_fd_sc_hd__inv_1 _238_ (.A(\u_tars.count[0] ),
    .Y(_000_));
 sky130_fd_sc_hd__nand3_1 _239_ (.A(net38),
    .B(lrc_ready),
    .C(sched_valid),
    .Y(_002_));
 sky130_fd_sc_hd__inv_1 _240_ (.A(_002_),
    .Y(_005_));
 sky130_fd_sc_hd__inv_1 _241_ (.A(\u_tars.count[1] ),
    .Y(_001_));
 sky130_fd_sc_hd__inv_1 _242_ (.A(\u_lrc.counter[1] ),
    .Y(_015_));
 sky130_fd_sc_hd__inv_1 _243_ (.A(\g_rfu[0].u_rfu.remaining[0] ),
    .Y(_008_));
 sky130_fd_sc_hd__inv_1 _244_ (.A(\g_rfu[0].u_rfu.remaining[1] ),
    .Y(_009_));
 sky130_fd_sc_hd__nor3_1 _245_ (.A(\g_rfu[0].u_rfu.remaining[5] ),
    .B(\g_rfu[0].u_rfu.remaining[3] ),
    .C(\g_rfu[0].u_rfu.remaining[4] ),
    .Y(_063_));
 sky130_fd_sc_hd__nor3b_1 _246_ (.A(\g_rfu[0].u_rfu.remaining[6] ),
    .B(\g_rfu[0].u_rfu.remaining[2] ),
    .C_N(_010_),
    .Y(_064_));
 sky130_fd_sc_hd__nand2_1 _247_ (.A(_063_),
    .B(_064_),
    .Y(_065_));
 sky130_fd_sc_hd__nor3_1 _248_ (.A(\g_rfu[0].u_rfu.remaining[10] ),
    .B(\g_rfu[0].u_rfu.remaining[11] ),
    .C(\g_rfu[0].u_rfu.remaining[12] ),
    .Y(_066_));
 sky130_fd_sc_hd__nor3_1 _249_ (.A(\g_rfu[0].u_rfu.remaining[15] ),
    .B(\g_rfu[0].u_rfu.remaining[13] ),
    .C(\g_rfu[0].u_rfu.remaining[14] ),
    .Y(_067_));
 sky130_fd_sc_hd__nor3_1 _250_ (.A(\g_rfu[0].u_rfu.remaining[7] ),
    .B(\g_rfu[0].u_rfu.remaining[8] ),
    .C(\g_rfu[0].u_rfu.remaining[9] ),
    .Y(_068_));
 sky130_fd_sc_hd__nand3_1 _251_ (.A(_066_),
    .B(_067_),
    .C(_068_),
    .Y(_069_));
 sky130_fd_sc_hd__and2_1 _253_ (.A(net38),
    .B(\g_rfu[0].u_rfu.issue_valid ),
    .X(_071_));
 sky130_fd_sc_hd__nand4bb_1 _256_ (.A_N(\g_rfu[0].u_rfu.remaining[2] ),
    .B_N(\g_rfu[0].u_rfu.remaining[3] ),
    .C(_010_),
    .D(\rfu_busy[0][0] ),
    .Y(_074_));
 sky130_fd_sc_hd__nor4_1 _257_ (.A(\g_rfu[0].u_rfu.remaining[4] ),
    .B(net41),
    .C(_071_),
    .D(_074_),
    .Y(_075_));
 sky130_fd_sc_hd__o21ai_0 _258_ (.A1(_065_),
    .A2(_069_),
    .B1(_075_),
    .Y(_076_));
 sky130_fd_sc_hd__inv_2 _259_ (.A(net1),
    .Y(_077_));
 sky130_fd_sc_hd__nand2_1 _260_ (.A(net38),
    .B(\g_rfu[0].u_rfu.issue_valid ),
    .Y(_078_));
 sky130_fd_sc_hd__and3_1 _262_ (.A(net12),
    .B(net40),
    .C(_071_),
    .X(_080_));
 sky130_fd_sc_hd__a41oi_1 _263_ (.A1(\g_rfu[0].u_rfu.remaining[4] ),
    .A2(net40),
    .A3(net37),
    .A4(_074_),
    .B1(_080_),
    .Y(_081_));
 sky130_fd_sc_hd__nand2_1 _264_ (.A(_076_),
    .B(_081_),
    .Y(_019_));
 sky130_fd_sc_hd__or4b_4 _266_ (.A(\g_rfu[0].u_rfu.remaining[0] ),
    .B(\g_rfu[0].u_rfu.remaining[1] ),
    .C(\g_rfu[0].u_rfu.remaining[2] ),
    .D_N(\rfu_busy[0][0] ),
    .X(_083_));
 sky130_fd_sc_hd__nand2_1 _267_ (.A(\g_rfu[0].u_rfu.remaining[3] ),
    .B(_083_),
    .Y(_084_));
 sky130_fd_sc_hd__nor2_1 _268_ (.A(\g_rfu[0].u_rfu.remaining[3] ),
    .B(_083_),
    .Y(_085_));
 sky130_fd_sc_hd__o21ai_0 _269_ (.A1(_065_),
    .A2(_069_),
    .B1(_085_),
    .Y(_086_));
 sky130_fd_sc_hd__o21ai_0 _271_ (.A1(net11),
    .A2(net37),
    .B1(net40),
    .Y(_088_));
 sky130_fd_sc_hd__a31oi_1 _272_ (.A1(net37),
    .A2(_084_),
    .A3(_086_),
    .B1(_088_),
    .Y(_020_));
 sky130_fd_sc_hd__nand2_1 _273_ (.A(_010_),
    .B(\rfu_busy[0][0] ),
    .Y(_089_));
 sky130_fd_sc_hd__nand2_1 _274_ (.A(\g_rfu[0].u_rfu.remaining[2] ),
    .B(_089_),
    .Y(_090_));
 sky130_fd_sc_hd__inv_1 _275_ (.A(\g_rfu[0].u_rfu.remaining[2] ),
    .Y(_091_));
 sky130_fd_sc_hd__o2111ai_1 _276_ (.A1(_065_),
    .A2(_069_),
    .B1(_091_),
    .C1(_010_),
    .D1(\rfu_busy[0][0] ),
    .Y(_092_));
 sky130_fd_sc_hd__nor2_1 _277_ (.A(net10),
    .B(net37),
    .Y(_093_));
 sky130_fd_sc_hd__a311oi_1 _279_ (.A1(net37),
    .A2(_090_),
    .A3(_092_),
    .B1(_093_),
    .C1(net41),
    .Y(_021_));
 sky130_fd_sc_hd__o21ai_0 _280_ (.A1(_065_),
    .A2(_069_),
    .B1(\rfu_busy[0][0] ),
    .Y(_095_));
 sky130_fd_sc_hd__o2111a_1 _281_ (.A1(_065_),
    .A2(_069_),
    .B1(_011_),
    .C1(\rfu_busy[0][0] ),
    .D1(net37),
    .X(_096_));
 sky130_fd_sc_hd__o21ai_0 _282_ (.A1(net9),
    .A2(net37),
    .B1(net40),
    .Y(_097_));
 sky130_fd_sc_hd__a311oi_1 _283_ (.A1(_009_),
    .A2(net37),
    .A3(_095_),
    .B1(_096_),
    .C1(_097_),
    .Y(_022_));
 sky130_fd_sc_hd__and4_1 _284_ (.A(_008_),
    .B(\rfu_busy[0][0] ),
    .C(net40),
    .D(net37),
    .X(_098_));
 sky130_fd_sc_hd__o21ai_0 _285_ (.A1(_065_),
    .A2(_069_),
    .B1(_098_),
    .Y(_099_));
 sky130_fd_sc_hd__nand3_1 _286_ (.A(\g_rfu[0].u_rfu.remaining[0] ),
    .B(net40),
    .C(net37),
    .Y(_100_));
 sky130_fd_sc_hd__or3_1 _287_ (.A(_065_),
    .B(_069_),
    .C(_100_),
    .X(_101_));
 sky130_fd_sc_hd__nor4_1 _288_ (.A(_008_),
    .B(\rfu_busy[0][0] ),
    .C(net41),
    .D(_071_),
    .Y(_102_));
 sky130_fd_sc_hd__a31oi_1 _289_ (.A1(net2),
    .A2(net40),
    .A3(_071_),
    .B1(_102_),
    .Y(_103_));
 sky130_fd_sc_hd__nand3_1 _290_ (.A(_099_),
    .B(_101_),
    .C(_103_),
    .Y(_023_));
 sky130_fd_sc_hd__nand2_1 _291_ (.A(net28),
    .B(net26),
    .Y(_104_));
 sky130_fd_sc_hd__nand2_1 _292_ (.A(_002_),
    .B(_104_),
    .Y(_105_));
 sky130_fd_sc_hd__nor2_1 _293_ (.A(\u_tars.count[1] ),
    .B(_105_),
    .Y(_106_));
 sky130_fd_sc_hd__a211oi_1 _294_ (.A1(_004_),
    .A2(_105_),
    .B1(_106_),
    .C1(net42),
    .Y(_024_));
 sky130_fd_sc_hd__a21oi_1 _295_ (.A1(net28),
    .A2(net26),
    .B1(_005_),
    .Y(_107_));
 sky130_fd_sc_hd__xnor2_1 _296_ (.A(_000_),
    .B(_107_),
    .Y(_108_));
 sky130_fd_sc_hd__nor2_1 _297_ (.A(net42),
    .B(_108_),
    .Y(_025_));
 sky130_fd_sc_hd__a21boi_0 _298_ (.A1(lrc_ready),
    .A2(sched_valid),
    .B1_N(\u_lrc.busy ),
    .Y(_109_));
 sky130_fd_sc_hd__or4b_2 _299_ (.A(\u_lrc.counter[7] ),
    .B(\u_lrc.counter[5] ),
    .C(\u_lrc.counter[6] ),
    .D_N(_016_),
    .X(_110_));
 sky130_fd_sc_hd__nor3_1 _301_ (.A(\u_lrc.counter[2] ),
    .B(\u_lrc.counter[3] ),
    .C(\u_lrc.counter[4] ),
    .Y(_112_));
 sky130_fd_sc_hd__nand3_1 _302_ (.A(_109_),
    .B(_110_),
    .C(_112_),
    .Y(_113_));
 sky130_fd_sc_hd__nand2b_1 _303_ (.A_N(\u_lrc.counter[5] ),
    .B(_018_),
    .Y(_114_));
 sky130_fd_sc_hd__o21ai_0 _304_ (.A1(_113_),
    .A2(_114_),
    .B1(\u_lrc.counter[6] ),
    .Y(_115_));
 sky130_fd_sc_hd__or3_1 _305_ (.A(\u_lrc.counter[6] ),
    .B(_113_),
    .C(_114_),
    .X(_116_));
 sky130_fd_sc_hd__xor2_1 _306_ (.A(\active_template[5] ),
    .B(net23),
    .X(_117_));
 sky130_fd_sc_hd__xor2_1 _307_ (.A(\active_template[0] ),
    .B(net18),
    .X(_118_));
 sky130_fd_sc_hd__xor2_1 _308_ (.A(\active_template[6] ),
    .B(net24),
    .X(_119_));
 sky130_fd_sc_hd__xor2_1 _309_ (.A(\active_template[4] ),
    .B(net22),
    .X(_120_));
 sky130_fd_sc_hd__nor4_1 _310_ (.A(_117_),
    .B(_118_),
    .C(_119_),
    .D(_120_),
    .Y(_121_));
 sky130_fd_sc_hd__xor2_1 _311_ (.A(\active_template[3] ),
    .B(net21),
    .X(_122_));
 sky130_fd_sc_hd__xor2_1 _312_ (.A(\active_template[2] ),
    .B(net20),
    .X(_123_));
 sky130_fd_sc_hd__xor2_1 _313_ (.A(\active_template[7] ),
    .B(net25),
    .X(_124_));
 sky130_fd_sc_hd__xor2_1 _314_ (.A(\active_template[1] ),
    .B(net19),
    .X(_125_));
 sky130_fd_sc_hd__nor4_1 _315_ (.A(_122_),
    .B(_123_),
    .C(_124_),
    .D(_125_),
    .Y(_126_));
 sky130_fd_sc_hd__nand2_1 _316_ (.A(lrc_ready),
    .B(sched_valid),
    .Y(_127_));
 sky130_fd_sc_hd__a21oi_1 _317_ (.A1(_121_),
    .A2(_126_),
    .B1(_127_),
    .Y(_128_));
 sky130_fd_sc_hd__a211oi_1 _319_ (.A1(_115_),
    .A2(_116_),
    .B1(net42),
    .C1(net36),
    .Y(_026_));
 sky130_fd_sc_hd__nor2_1 _320_ (.A(\u_lrc.counter[0] ),
    .B(\u_lrc.counter[1] ),
    .Y(_130_));
 sky130_fd_sc_hd__nand4_1 _321_ (.A(_109_),
    .B(_110_),
    .C(_112_),
    .D(_130_),
    .Y(_131_));
 sky130_fd_sc_hd__xor2_1 _322_ (.A(\u_lrc.counter[5] ),
    .B(_131_),
    .X(_132_));
 sky130_fd_sc_hd__nor3_1 _323_ (.A(net42),
    .B(net36),
    .C(_132_),
    .Y(_027_));
 sky130_fd_sc_hd__nor2_1 _324_ (.A(\u_lrc.counter[2] ),
    .B(\u_lrc.counter[3] ),
    .Y(_133_));
 sky130_fd_sc_hd__o2111ai_1 _325_ (.A1(\u_lrc.counter[4] ),
    .A2(_110_),
    .B1(_109_),
    .C1(_133_),
    .D1(_018_),
    .Y(_134_));
 sky130_fd_sc_hd__xnor2_1 _326_ (.A(\u_lrc.counter[4] ),
    .B(_134_),
    .Y(_135_));
 sky130_fd_sc_hd__o21a_1 _328_ (.A1(net36),
    .A2(_135_),
    .B1(net40),
    .X(_028_));
 sky130_fd_sc_hd__or3_1 _329_ (.A(\u_lrc.counter[2] ),
    .B(\u_lrc.counter[3] ),
    .C(\u_lrc.counter[4] ),
    .X(_137_));
 sky130_fd_sc_hd__nor3_1 _330_ (.A(\u_lrc.counter[0] ),
    .B(\u_lrc.counter[1] ),
    .C(\u_lrc.counter[2] ),
    .Y(_138_));
 sky130_fd_sc_hd__o211ai_1 _331_ (.A1(_110_),
    .A2(_137_),
    .B1(_138_),
    .C1(_109_),
    .Y(_139_));
 sky130_fd_sc_hd__xor2_1 _332_ (.A(\u_lrc.counter[3] ),
    .B(_139_),
    .X(_140_));
 sky130_fd_sc_hd__nor3_1 _333_ (.A(net42),
    .B(net36),
    .C(_140_),
    .Y(_029_));
 sky130_fd_sc_hd__nand2_1 _334_ (.A(\u_tars.count[3] ),
    .B(net39),
    .Y(_030_));
 sky130_fd_sc_hd__or2_2 _335_ (.A(\u_tars.count[3] ),
    .B(net42),
    .X(_141_));
 sky130_fd_sc_hd__nand3_1 _336_ (.A(net28),
    .B(net26),
    .C(_002_),
    .Y(_142_));
 sky130_fd_sc_hd__a21o_1 _337_ (.A1(\u_tars.count[0] ),
    .A2(_013_),
    .B1(_012_),
    .X(_143_));
 sky130_fd_sc_hd__a21oi_1 _338_ (.A1(_007_),
    .A2(_143_),
    .B1(_006_),
    .Y(_144_));
 sky130_fd_sc_hd__mux2i_1 _339_ (.A0(_142_),
    .A1(_002_),
    .S(_144_),
    .Y(_145_));
 sky130_fd_sc_hd__mux2i_1 _340_ (.A0(_030_),
    .A1(_141_),
    .S(_145_),
    .Y(_031_));
 sky130_fd_sc_hd__nor4_1 _341_ (.A(\u_tars.count[2] ),
    .B(\u_tars.count[3] ),
    .C(\u_tars.count[0] ),
    .D(\u_tars.count[1] ),
    .Y(_146_));
 sky130_fd_sc_hd__nor2_1 _342_ (.A(net42),
    .B(_146_),
    .Y(_032_));
 sky130_fd_sc_hd__o211ai_1 _343_ (.A1(_110_),
    .A2(_137_),
    .B1(_018_),
    .C1(_109_),
    .Y(_147_));
 sky130_fd_sc_hd__xor2_1 _344_ (.A(\u_lrc.counter[2] ),
    .B(_147_),
    .X(_148_));
 sky130_fd_sc_hd__nor3_1 _345_ (.A(net42),
    .B(net36),
    .C(_148_),
    .Y(_033_));
 sky130_fd_sc_hd__o21ai_0 _346_ (.A1(_110_),
    .A2(_137_),
    .B1(_109_),
    .Y(_149_));
 sky130_fd_sc_hd__nand3_1 _347_ (.A(\u_lrc.counter[1] ),
    .B(net39),
    .C(_149_),
    .Y(_150_));
 sky130_fd_sc_hd__o32ai_1 _348_ (.A1(_017_),
    .A2(net42),
    .A3(_149_),
    .B1(_150_),
    .B2(net36),
    .Y(_034_));
 sky130_fd_sc_hd__nand3_1 _349_ (.A(\u_lrc.counter[0] ),
    .B(net39),
    .C(_149_),
    .Y(_151_));
 sky130_fd_sc_hd__or3_1 _350_ (.A(\u_lrc.counter[0] ),
    .B(net42),
    .C(_149_),
    .X(_152_));
 sky130_fd_sc_hd__o21ai_0 _351_ (.A1(net36),
    .A2(_151_),
    .B1(_152_),
    .Y(_035_));
 sky130_fd_sc_hd__nand2_1 _352_ (.A(\u_tars.count[2] ),
    .B(_107_),
    .Y(_153_));
 sky130_fd_sc_hd__xnor2_1 _353_ (.A(_007_),
    .B(_003_),
    .Y(_154_));
 sky130_fd_sc_hd__nand2_1 _354_ (.A(_105_),
    .B(_154_),
    .Y(_155_));
 sky130_fd_sc_hd__a21oi_1 _355_ (.A1(_153_),
    .A2(_155_),
    .B1(net42),
    .Y(_036_));
 sky130_fd_sc_hd__nand3_1 _356_ (.A(\rfu_busy[0][0] ),
    .B(_063_),
    .C(_064_),
    .Y(_156_));
 sky130_fd_sc_hd__nand2_1 _357_ (.A(_066_),
    .B(_068_),
    .Y(_157_));
 sky130_fd_sc_hd__nand2b_1 _358_ (.A_N(\g_rfu[0].u_rfu.remaining[14] ),
    .B(\g_rfu[0].u_rfu.remaining[15] ),
    .Y(_158_));
 sky130_fd_sc_hd__or4_1 _359_ (.A(\g_rfu[0].u_rfu.remaining[13] ),
    .B(_156_),
    .C(_157_),
    .D(_158_),
    .X(_159_));
 sky130_fd_sc_hd__o31ai_1 _360_ (.A1(\g_rfu[0].u_rfu.remaining[13] ),
    .A2(_156_),
    .A3(_157_),
    .B1(\g_rfu[0].u_rfu.remaining[14] ),
    .Y(_160_));
 sky130_fd_sc_hd__o21ai_0 _361_ (.A1(net7),
    .A2(net37),
    .B1(net40),
    .Y(_161_));
 sky130_fd_sc_hd__a31oi_1 _362_ (.A1(net37),
    .A2(_159_),
    .A3(_160_),
    .B1(_161_),
    .Y(_037_));
 sky130_fd_sc_hd__and4_1 _363_ (.A(_063_),
    .B(_064_),
    .C(_066_),
    .D(_067_),
    .X(_162_));
 sky130_fd_sc_hd__or3_1 _364_ (.A(\g_rfu[0].u_rfu.remaining[10] ),
    .B(\g_rfu[0].u_rfu.remaining[11] ),
    .C(\g_rfu[0].u_rfu.remaining[12] ),
    .X(_163_));
 sky130_fd_sc_hd__or3_1 _365_ (.A(\g_rfu[0].u_rfu.remaining[7] ),
    .B(\g_rfu[0].u_rfu.remaining[8] ),
    .C(\g_rfu[0].u_rfu.remaining[9] ),
    .X(_164_));
 sky130_fd_sc_hd__or4_1 _366_ (.A(\g_rfu[0].u_rfu.remaining[5] ),
    .B(\g_rfu[0].u_rfu.remaining[6] ),
    .C(\g_rfu[0].u_rfu.remaining[3] ),
    .D(\g_rfu[0].u_rfu.remaining[4] ),
    .X(_165_));
 sky130_fd_sc_hd__or4_4 _367_ (.A(_163_),
    .B(_164_),
    .C(_083_),
    .D(_165_),
    .X(_166_));
 sky130_fd_sc_hd__o21ai_0 _368_ (.A1(_162_),
    .A2(_166_),
    .B1(\g_rfu[0].u_rfu.remaining[13] ),
    .Y(_167_));
 sky130_fd_sc_hd__or3_1 _369_ (.A(\g_rfu[0].u_rfu.remaining[13] ),
    .B(_162_),
    .C(_166_),
    .X(_168_));
 sky130_fd_sc_hd__o21ai_0 _370_ (.A1(net6),
    .A2(net37),
    .B1(net40),
    .Y(_169_));
 sky130_fd_sc_hd__a31oi_1 _371_ (.A1(net37),
    .A2(_167_),
    .A3(_168_),
    .B1(_169_),
    .Y(_038_));
 sky130_fd_sc_hd__nand3_1 _372_ (.A(net37),
    .B(_063_),
    .C(_064_),
    .Y(_170_));
 sky130_fd_sc_hd__o221ai_1 _373_ (.A1(\rfu_busy[0][0] ),
    .A2(_071_),
    .B1(_069_),
    .B2(_170_),
    .C1(net40),
    .Y(_171_));
 sky130_fd_sc_hd__nand2_1 _374_ (.A(net5),
    .B(_071_),
    .Y(_172_));
 sky130_fd_sc_hd__o31a_1 _375_ (.A1(_071_),
    .A2(_065_),
    .A3(_157_),
    .B1(_172_),
    .X(_173_));
 sky130_fd_sc_hd__nand4_1 _376_ (.A(\rfu_busy[0][0] ),
    .B(_063_),
    .C(_064_),
    .D(_068_),
    .Y(_174_));
 sky130_fd_sc_hd__nor2_1 _377_ (.A(net41),
    .B(_071_),
    .Y(_175_));
 sky130_fd_sc_hd__o311ai_0 _378_ (.A1(\g_rfu[0].u_rfu.remaining[10] ),
    .A2(\g_rfu[0].u_rfu.remaining[11] ),
    .A3(_174_),
    .B1(_175_),
    .C1(\g_rfu[0].u_rfu.remaining[12] ),
    .Y(_176_));
 sky130_fd_sc_hd__o21ai_0 _379_ (.A1(_171_),
    .A2(_173_),
    .B1(_176_),
    .Y(_039_));
 sky130_fd_sc_hd__or3_1 _380_ (.A(\g_rfu[0].u_rfu.remaining[5] ),
    .B(\g_rfu[0].u_rfu.remaining[3] ),
    .C(\g_rfu[0].u_rfu.remaining[4] ),
    .X(_177_));
 sky130_fd_sc_hd__or3_1 _381_ (.A(\g_rfu[0].u_rfu.remaining[6] ),
    .B(_177_),
    .C(_083_),
    .X(_178_));
 sky130_fd_sc_hd__o41ai_1 _382_ (.A1(\g_rfu[0].u_rfu.remaining[10] ),
    .A2(_164_),
    .A3(_162_),
    .A4(_178_),
    .B1(\g_rfu[0].u_rfu.remaining[11] ),
    .Y(_179_));
 sky130_fd_sc_hd__nor2_1 _383_ (.A(\g_rfu[0].u_rfu.remaining[10] ),
    .B(_164_),
    .Y(_180_));
 sky130_fd_sc_hd__or4b_1 _384_ (.A(\g_rfu[0].u_rfu.remaining[11] ),
    .B(_162_),
    .C(_178_),
    .D_N(_180_),
    .X(_181_));
 sky130_fd_sc_hd__o21ai_0 _385_ (.A1(net4),
    .A2(net37),
    .B1(net40),
    .Y(_182_));
 sky130_fd_sc_hd__a31oi_1 _386_ (.A1(net37),
    .A2(_179_),
    .A3(_181_),
    .B1(_182_),
    .Y(_040_));
 sky130_fd_sc_hd__nand2_1 _387_ (.A(_066_),
    .B(_067_),
    .Y(_183_));
 sky130_fd_sc_hd__or3b_2 _388_ (.A(\g_rfu[0].u_rfu.remaining[10] ),
    .B(_174_),
    .C_N(_183_),
    .X(_184_));
 sky130_fd_sc_hd__nand2_1 _389_ (.A(\g_rfu[0].u_rfu.remaining[10] ),
    .B(_174_),
    .Y(_185_));
 sky130_fd_sc_hd__nor2_1 _390_ (.A(net3),
    .B(net37),
    .Y(_186_));
 sky130_fd_sc_hd__a311oi_1 _391_ (.A1(net37),
    .A2(_184_),
    .A3(_185_),
    .B1(_186_),
    .C1(net41),
    .Y(_041_));
 sky130_fd_sc_hd__nand2_1 _392_ (.A(\active_template[6] ),
    .B(net39),
    .Y(_187_));
 sky130_fd_sc_hd__nand2_1 _394_ (.A(net24),
    .B(net39),
    .Y(_189_));
 sky130_fd_sc_hd__mux2i_1 _395_ (.A0(_187_),
    .A1(_189_),
    .S(net36),
    .Y(_042_));
 sky130_fd_sc_hd__nand2_1 _396_ (.A(\active_template[5] ),
    .B(net39),
    .Y(_190_));
 sky130_fd_sc_hd__nand2_1 _397_ (.A(net23),
    .B(net39),
    .Y(_191_));
 sky130_fd_sc_hd__mux2i_1 _398_ (.A0(_190_),
    .A1(_191_),
    .S(net36),
    .Y(_043_));
 sky130_fd_sc_hd__nand2_1 _399_ (.A(\active_template[4] ),
    .B(net39),
    .Y(_192_));
 sky130_fd_sc_hd__nand2_1 _400_ (.A(net22),
    .B(net39),
    .Y(_193_));
 sky130_fd_sc_hd__mux2i_1 _401_ (.A0(_192_),
    .A1(_193_),
    .S(net36),
    .Y(_044_));
 sky130_fd_sc_hd__nand2_1 _402_ (.A(\active_template[3] ),
    .B(net39),
    .Y(_194_));
 sky130_fd_sc_hd__nand2_1 _403_ (.A(net21),
    .B(net39),
    .Y(_195_));
 sky130_fd_sc_hd__mux2i_1 _404_ (.A0(_194_),
    .A1(_195_),
    .S(net36),
    .Y(_045_));
 sky130_fd_sc_hd__o31ai_1 _405_ (.A1(\u_lrc.counter[5] ),
    .A2(\u_lrc.counter[6] ),
    .A3(_131_),
    .B1(\u_lrc.counter[7] ),
    .Y(_196_));
 sky130_fd_sc_hd__or4_1 _406_ (.A(\u_lrc.counter[7] ),
    .B(\u_lrc.counter[5] ),
    .C(\u_lrc.counter[6] ),
    .D(_131_),
    .X(_197_));
 sky130_fd_sc_hd__a211oi_1 _407_ (.A1(_196_),
    .A2(_197_),
    .B1(net42),
    .C1(net36),
    .Y(_046_));
 sky130_fd_sc_hd__nand2_1 _408_ (.A(_121_),
    .B(_126_),
    .Y(_198_));
 sky130_fd_sc_hd__a21oi_1 _409_ (.A1(_127_),
    .A2(_149_),
    .B1(net42),
    .Y(_199_));
 sky130_fd_sc_hd__o21a_1 _410_ (.A1(\u_lrc.busy ),
    .A2(_198_),
    .B1(_199_),
    .X(_047_));
 sky130_fd_sc_hd__nor2_1 _411_ (.A(_110_),
    .B(_137_),
    .Y(_200_));
 sky130_fd_sc_hd__a21oi_1 _412_ (.A1(\u_lrc.busy ),
    .A2(_200_),
    .B1(lrc_ready),
    .Y(_201_));
 sky130_fd_sc_hd__o21ai_0 _413_ (.A1(net36),
    .A2(_201_),
    .B1(net39),
    .Y(_048_));
 sky130_fd_sc_hd__nand2_1 _414_ (.A(\active_template[7] ),
    .B(net39),
    .Y(_202_));
 sky130_fd_sc_hd__nand2_1 _415_ (.A(net25),
    .B(net39),
    .Y(_203_));
 sky130_fd_sc_hd__mux2i_1 _416_ (.A0(_202_),
    .A1(_203_),
    .S(net36),
    .Y(_049_));
 sky130_fd_sc_hd__nand4_1 _417_ (.A(lrc_ready),
    .B(sched_valid),
    .C(_121_),
    .D(_126_),
    .Y(_204_));
 sky130_fd_sc_hd__nand3_1 _418_ (.A(\u_lrc.busy ),
    .B(_127_),
    .C(_200_),
    .Y(_205_));
 sky130_fd_sc_hd__a21oi_1 _419_ (.A1(_204_),
    .A2(_205_),
    .B1(net42),
    .Y(_050_));
 sky130_fd_sc_hd__nand2_1 _420_ (.A(\active_template[2] ),
    .B(net39),
    .Y(_206_));
 sky130_fd_sc_hd__nand2_1 _421_ (.A(net20),
    .B(net39),
    .Y(_207_));
 sky130_fd_sc_hd__mux2i_1 _422_ (.A0(_206_),
    .A1(_207_),
    .S(net36),
    .Y(_051_));
 sky130_fd_sc_hd__nand2_1 _423_ (.A(\active_template[1] ),
    .B(net39),
    .Y(_208_));
 sky130_fd_sc_hd__nand2_1 _424_ (.A(net19),
    .B(net39),
    .Y(_209_));
 sky130_fd_sc_hd__mux2i_1 _425_ (.A0(_208_),
    .A1(_209_),
    .S(net36),
    .Y(_052_));
 sky130_fd_sc_hd__nand2_1 _426_ (.A(\active_template[0] ),
    .B(net39),
    .Y(_210_));
 sky130_fd_sc_hd__nand2_1 _427_ (.A(net18),
    .B(net39),
    .Y(_211_));
 sky130_fd_sc_hd__mux2i_1 _428_ (.A0(_210_),
    .A1(_211_),
    .S(net36),
    .Y(_053_));
 sky130_fd_sc_hd__nor2_1 _429_ (.A(_083_),
    .B(_165_),
    .Y(_212_));
 sky130_fd_sc_hd__nor2_1 _430_ (.A(\g_rfu[0].u_rfu.remaining[7] ),
    .B(\g_rfu[0].u_rfu.remaining[8] ),
    .Y(_213_));
 sky130_fd_sc_hd__inv_1 _431_ (.A(\g_rfu[0].u_rfu.remaining[9] ),
    .Y(_214_));
 sky130_fd_sc_hd__o2111ai_1 _432_ (.A1(_065_),
    .A2(_069_),
    .B1(_212_),
    .C1(_213_),
    .D1(_214_),
    .Y(_215_));
 sky130_fd_sc_hd__a21o_1 _433_ (.A1(_213_),
    .A2(_212_),
    .B1(_214_),
    .X(_216_));
 sky130_fd_sc_hd__o21ai_0 _434_ (.A1(net17),
    .A2(net37),
    .B1(net40),
    .Y(_217_));
 sky130_fd_sc_hd__a31oi_1 _435_ (.A1(net37),
    .A2(_215_),
    .A3(_216_),
    .B1(_217_),
    .Y(_054_));
 sky130_fd_sc_hd__inv_1 _436_ (.A(_171_),
    .Y(_055_));
 sky130_fd_sc_hd__a31oi_1 _437_ (.A1(_214_),
    .A2(_066_),
    .A3(_067_),
    .B1(\g_rfu[0].u_rfu.remaining[8] ),
    .Y(_218_));
 sky130_fd_sc_hd__nand4b_1 _438_ (.A_N(\g_rfu[0].u_rfu.remaining[7] ),
    .B(\rfu_busy[0][0] ),
    .C(_063_),
    .D(_064_),
    .Y(_219_));
 sky130_fd_sc_hd__mux2i_1 _439_ (.A0(_218_),
    .A1(\g_rfu[0].u_rfu.remaining[8] ),
    .S(_219_),
    .Y(_220_));
 sky130_fd_sc_hd__nor2_1 _440_ (.A(net16),
    .B(net37),
    .Y(_221_));
 sky130_fd_sc_hd__a211oi_1 _441_ (.A1(net37),
    .A2(_220_),
    .B1(_221_),
    .C1(net41),
    .Y(_056_));
 sky130_fd_sc_hd__nand2b_1 _442_ (.A_N(\g_rfu[0].u_rfu.issue_valid ),
    .B(net38),
    .Y(_222_));
 sky130_fd_sc_hd__o311ai_0 _443_ (.A1(net38),
    .A2(_183_),
    .A3(_174_),
    .B1(_222_),
    .C1(_077_),
    .Y(_057_));
 sky130_fd_sc_hd__and3_1 _444_ (.A(_066_),
    .B(_067_),
    .C(_068_),
    .X(_223_));
 sky130_fd_sc_hd__a311o_1 _445_ (.A1(_063_),
    .A2(_064_),
    .A3(_223_),
    .B1(_178_),
    .C1(\g_rfu[0].u_rfu.remaining[7] ),
    .X(_224_));
 sky130_fd_sc_hd__a21oi_1 _446_ (.A1(\g_rfu[0].u_rfu.remaining[7] ),
    .A2(_178_),
    .B1(_071_),
    .Y(_225_));
 sky130_fd_sc_hd__o21ai_0 _447_ (.A1(net15),
    .A2(net37),
    .B1(net40),
    .Y(_226_));
 sky130_fd_sc_hd__a21oi_1 _448_ (.A1(_224_),
    .A2(_225_),
    .B1(_226_),
    .Y(_058_));
 sky130_fd_sc_hd__o31ai_1 _449_ (.A1(\g_rfu[0].u_rfu.remaining[2] ),
    .A2(_177_),
    .A3(_089_),
    .B1(\g_rfu[0].u_rfu.remaining[6] ),
    .Y(_227_));
 sky130_fd_sc_hd__o211ai_1 _450_ (.A1(_223_),
    .A2(_156_),
    .B1(_227_),
    .C1(net37),
    .Y(_228_));
 sky130_fd_sc_hd__o211a_1 _451_ (.A1(net14),
    .A2(net37),
    .B1(_228_),
    .C1(net40),
    .X(_059_));
 sky130_fd_sc_hd__nor3_1 _452_ (.A(\g_rfu[0].u_rfu.remaining[0] ),
    .B(\g_rfu[0].u_rfu.remaining[1] ),
    .C(_071_),
    .Y(_229_));
 sky130_fd_sc_hd__a32oi_1 _453_ (.A1(_091_),
    .A2(_063_),
    .A3(_229_),
    .B1(net13),
    .B2(_071_),
    .Y(_230_));
 sky130_fd_sc_hd__or2_2 _454_ (.A(\g_rfu[0].u_rfu.remaining[3] ),
    .B(\g_rfu[0].u_rfu.remaining[4] ),
    .X(_231_));
 sky130_fd_sc_hd__o211ai_1 _455_ (.A1(_231_),
    .A2(_083_),
    .B1(\g_rfu[0].u_rfu.remaining[5] ),
    .C1(_175_),
    .Y(_232_));
 sky130_fd_sc_hd__o21ai_0 _456_ (.A1(_171_),
    .A2(_230_),
    .B1(_232_),
    .Y(_060_));
 sky130_fd_sc_hd__nor4_1 _457_ (.A(net41),
    .B(_071_),
    .C(_183_),
    .D(_174_),
    .Y(_061_));
 sky130_fd_sc_hd__or2_2 _458_ (.A(\g_rfu[0].u_rfu.remaining[13] ),
    .B(\g_rfu[0].u_rfu.remaining[14] ),
    .X(_233_));
 sky130_fd_sc_hd__or4_1 _459_ (.A(\g_rfu[0].u_rfu.remaining[15] ),
    .B(_233_),
    .C(_162_),
    .D(_166_),
    .X(_234_));
 sky130_fd_sc_hd__o31ai_1 _460_ (.A1(_233_),
    .A2(_162_),
    .A3(_166_),
    .B1(\g_rfu[0].u_rfu.remaining[15] ),
    .Y(_235_));
 sky130_fd_sc_hd__o21ai_0 _461_ (.A1(net8),
    .A2(net37),
    .B1(net40),
    .Y(_236_));
 sky130_fd_sc_hd__a31oi_1 _462_ (.A1(net37),
    .A2(_234_),
    .A3(_235_),
    .B1(_236_),
    .Y(_062_));
 sky130_fd_sc_hd__inv_1 _463_ (.A(\u_lrc.counter[0] ),
    .Y(_014_));
 sky130_fd_sc_hd__fa_1 _464_ (.A(_000_),
    .B(_001_),
    .CIN(_002_),
    .COUT(_003_),
    .SUM(_004_));
 sky130_fd_sc_hd__ha_1 _465_ (.A(\u_tars.count[2] ),
    .B(_005_),
    .COUT(_006_),
    .SUM(_007_));
 sky130_fd_sc_hd__ha_1 _466_ (.A(_008_),
    .B(_009_),
    .COUT(_010_),
    .SUM(_011_));
 sky130_fd_sc_hd__ha_1 _467_ (.A(\u_tars.count[1] ),
    .B(_005_),
    .COUT(_012_),
    .SUM(_013_));
 sky130_fd_sc_hd__ha_1 _468_ (.A(_014_),
    .B(_015_),
    .COUT(_016_),
    .SUM(_017_));
 sky130_fd_sc_hd__ha_1 _469_ (.A(_014_),
    .B(_015_),
    .COUT(_018_),
    .SUM(_237_));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_0_clk (.A(clk),
    .X(clknet_0_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_2_0_0_clk (.A(clknet_0_clk),
    .X(clknet_2_0_0_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_2_1_0_clk (.A(clknet_0_clk),
    .X(clknet_2_1_0_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_2_2_0_clk (.A(clknet_0_clk),
    .X(clknet_2_2_0_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_2_3_0_clk (.A(clknet_0_clk),
    .X(clknet_2_3_0_clk));
 sky130_fd_sc_hd__bufinv_16 clkload0 (.A(clknet_2_1_0_clk));
 sky130_fd_sc_hd__clkinvlp_4 clkload1 (.A(clknet_2_2_0_clk));
 sky130_fd_sc_hd__clkinv_2 clkload2 (.A(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.busy$_SDFF_PP0_  (.D(_055_),
    .Q(\rfu_busy[0][0] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.done$_SDFF_PP0_  (.D(_061_),
    .Q(net27),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.issue_ready$_SDFFE_PP1P_  (.D(_057_),
    .Q(\rfu_ready[0][0] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[0]$_SDFFE_PP0P_  (.D(_023_),
    .Q(\g_rfu[0].u_rfu.remaining[0] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[10]$_SDFFE_PP0P_  (.D(_041_),
    .Q(\g_rfu[0].u_rfu.remaining[10] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[11]$_SDFFE_PP0P_  (.D(_040_),
    .Q(\g_rfu[0].u_rfu.remaining[11] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[12]$_SDFFE_PP0P_  (.D(_039_),
    .Q(\g_rfu[0].u_rfu.remaining[12] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[13]$_SDFFE_PP0P_  (.D(_038_),
    .Q(\g_rfu[0].u_rfu.remaining[13] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[14]$_SDFFE_PP0P_  (.D(_037_),
    .Q(\g_rfu[0].u_rfu.remaining[14] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[15]$_SDFFE_PP0P_  (.D(_062_),
    .Q(\g_rfu[0].u_rfu.remaining[15] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[1]$_SDFFE_PP0P_  (.D(_022_),
    .Q(\g_rfu[0].u_rfu.remaining[1] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[2]$_SDFFE_PP0P_  (.D(_021_),
    .Q(\g_rfu[0].u_rfu.remaining[2] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[3]$_SDFFE_PP0P_  (.D(_020_),
    .Q(\g_rfu[0].u_rfu.remaining[3] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[4]$_SDFFE_PP0P_  (.D(_019_),
    .Q(\g_rfu[0].u_rfu.remaining[4] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[5]$_SDFFE_PP0P_  (.D(_060_),
    .Q(\g_rfu[0].u_rfu.remaining[5] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[6]$_SDFFE_PP0P_  (.D(_059_),
    .Q(\g_rfu[0].u_rfu.remaining[6] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[7]$_SDFFE_PP0P_  (.D(_058_),
    .Q(\g_rfu[0].u_rfu.remaining[7] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[8]$_SDFFE_PP0P_  (.D(_056_),
    .Q(\g_rfu[0].u_rfu.remaining[8] ),
    .CLK(clknet_2_3_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \g_rfu[0].u_rfu.remaining[9]$_SDFFE_PP0P_  (.D(_054_),
    .Q(\g_rfu[0].u_rfu.remaining[9] ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__buf_2 input1 (.A(rst),
    .X(net1));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input10 (.A(task_latency[2]),
    .X(net10));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input11 (.A(task_latency[3]),
    .X(net11));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input12 (.A(task_latency[4]),
    .X(net12));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input13 (.A(task_latency[5]),
    .X(net13));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input14 (.A(task_latency[6]),
    .X(net14));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input15 (.A(task_latency[7]),
    .X(net15));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input16 (.A(task_latency[8]),
    .X(net16));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input17 (.A(task_latency[9]),
    .X(net17));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input18 (.A(task_template[0]),
    .X(net18));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input19 (.A(task_template[1]),
    .X(net19));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input2 (.A(task_latency[0]),
    .X(net2));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input20 (.A(task_template[2]),
    .X(net20));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input21 (.A(task_template[3]),
    .X(net21));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input22 (.A(task_template[4]),
    .X(net22));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input23 (.A(task_template[5]),
    .X(net23));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input24 (.A(task_template[6]),
    .X(net24));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input25 (.A(task_template[7]),
    .X(net25));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input26 (.A(task_valid),
    .X(net26));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input3 (.A(task_latency[10]),
    .X(net3));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input4 (.A(task_latency[11]),
    .X(net4));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input5 (.A(task_latency[12]),
    .X(net5));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input6 (.A(task_latency[13]),
    .X(net6));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input7 (.A(task_latency[14]),
    .X(net7));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input8 (.A(task_latency[15]),
    .X(net8));
 sky130_fd_sc_hd__clkdlybuf4s50_1 input9 (.A(task_latency[1]),
    .X(net9));
 sky130_fd_sc_hd__buf_2 output28 (.A(net28),
    .X(task_ready));
 sky130_fd_sc_hd__buf_4 place31 (.A(net27),
    .X(done));
 sky130_fd_sc_hd__buf_4 place35 (.A(_050_),
    .X(net35));
 sky130_fd_sc_hd__buf_4 place36 (.A(_128_),
    .X(net36));
 sky130_fd_sc_hd__buf_4 place37 (.A(_078_),
    .X(net37));
 sky130_fd_sc_hd__buf_4 place38 (.A(\rfu_ready[0][0] ),
    .X(net38));
 sky130_fd_sc_hd__buf_4 place39 (.A(net40),
    .X(net39));
 sky130_fd_sc_hd__buf_4 place40 (.A(_077_),
    .X(net40));
 sky130_fd_sc_hd__buf_4 place41 (.A(net42),
    .X(net41));
 sky130_fd_sc_hd__buf_4 place42 (.A(net1),
    .X(net42));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[0]$_SDFFE_PP0P_  (.D(_053_),
    .Q(\active_template[0] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[1]$_SDFFE_PP0P_  (.D(_052_),
    .Q(\active_template[1] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[2]$_SDFFE_PP0P_  (.D(_051_),
    .Q(\active_template[2] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[3]$_SDFFE_PP0P_  (.D(_045_),
    .Q(\active_template[3] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[4]$_SDFFE_PP0P_  (.D(_044_),
    .Q(\active_template[4] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[5]$_SDFFE_PP0P_  (.D(_043_),
    .Q(\active_template[5] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[6]$_SDFFE_PP0P_  (.D(_042_),
    .Q(\active_template[6] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.active_template[7]$_SDFFE_PP0P_  (.D(_049_),
    .Q(\active_template[7] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.busy$_SDFFE_PP0P_  (.D(_047_),
    .Q(\u_lrc.busy ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[0]$_SDFFE_PP0P_  (.D(_035_),
    .Q(\u_lrc.counter[0] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[1]$_SDFFE_PP0P_  (.D(_034_),
    .Q(\u_lrc.counter[1] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[2]$_SDFFE_PP0P_  (.D(_033_),
    .Q(\u_lrc.counter[2] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[3]$_SDFFE_PP0P_  (.D(_029_),
    .Q(\u_lrc.counter[3] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[4]$_SDFFE_PP0P_  (.D(_028_),
    .Q(\u_lrc.counter[4] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[5]$_SDFFE_PP0P_  (.D(_027_),
    .Q(\u_lrc.counter[5] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[6]$_SDFFE_PP0P_  (.D(_026_),
    .Q(\u_lrc.counter[6] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.counter[7]$_SDFFE_PP0P_  (.D(_046_),
    .Q(\u_lrc.counter[7] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.done$_SDFF_PP0_  (.D(net35),
    .Q(\g_rfu[0].u_rfu.issue_valid ),
    .CLK(clknet_2_2_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_lrc.req_ready$_SDFFE_PP1P_  (.D(_048_),
    .Q(lrc_ready),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_tars.count[0]$_SDFFE_PP0P_  (.D(_025_),
    .Q(\u_tars.count[0] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_tars.count[1]$_SDFFE_PP0P_  (.D(_024_),
    .Q(\u_tars.count[1] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_tars.count[2]$_SDFFE_PP0P_  (.D(_036_),
    .Q(\u_tars.count[2] ),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_tars.count[3]$_SDFFE_PP0P_  (.D(_031_),
    .Q(\u_tars.count[3] ),
    .CLK(clknet_2_0_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_tars.pop_valid$_SDFF_PP0_  (.D(_032_),
    .Q(sched_valid),
    .CLK(clknet_2_1_0_clk));
 sky130_fd_sc_hd__dfxtp_1 \u_tars.push_ready$_SDFF_PP1_  (.D(_030_),
    .Q(net28),
    .CLK(clknet_2_1_0_clk));
endmodule
