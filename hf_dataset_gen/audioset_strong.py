import os
import datasets
import numpy as np
import pandas as pd
import h5py
import csv
from datasets import Value, Sequence


_CITATION = "@INPROCEEDINGS{9414579, author={Hershey, Shawn and Ellis, Daniel P W and Fonseca, Eduardo and Jansen, Aren and Liu, Caroline and Channing Moore, R and Plakal, Manoj}, booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, title={The Benefit of Temporally-Strong Labels in Audio Event Classification}, year={2021},volume={},number={},pages={366-370},keywords={Training;Conferences;Signal processing;Acoustics;Speech processing;AudioSet;audio event classification;explicit negatives;temporally-strong labels},doi={10.1109/ICASSP39728.2021.9414579}}"

_DESCRIPTION = "A detailed description of the dataset can be found here: https://research.google.com/audioset/download_strong.html++labels_strong++/g/11b630rrvh::Kettle whistle;;/g/122z_qxw::Firecracker;;/m/01280g::Wild animals;;/m/012f08::Motor vehicle (road);;/m/012n7d::Ambulance (siren);;/m/012ndj::Fire engine, fire truck (siren);;/m/012xff::Toothbrush;;/m/0130jx::Sink (filling or washing);;/m/014yck::Aircraft engine;;/m/014zdl::Explosion;;/m/0150b9::Change ringing (campanology);;/m/015jpf::Dial tone;;/m/015lz1::Singing;;/m/015p6::Bird;;/m/0160x5::Digestive;;/m/0174k2::Washing machine;;/m/018p4k::Cart;;/m/018w8::Basketball bounce;;/m/0193bn::Sonic boom;;/m/0195fx::Subway, metro, underground;;/m/0199g::Bicycle, tricycle;;/m/019jd::Boat, Water vehicle;;/m/01b82r::Sawing;;/m/01b9nn::Reverberation;;/m/01b_21::Cough;;/m/01bjv::Bus;;/m/01c194::Mantra;;/m/01d380::Drill;;/m/01d3sd::Snoring;;/m/01dwxx::Gull, seagull;;/m/01g50p::Railroad car, train wagon;;/m/01g90h::Stomach rumble;;/m/01h3n::Bee, wasp, etc.;;/m/01h82_::Engine knocking;;/m/01h8n0::Conversation;;/m/01hhp3::Yak;;/m/01hnzm::Ringtone;;/m/01hsr_::Sneeze;;/m/01j2bj::Bathroom sounds;;/m/01j3j8::Studio recording;;/m/01j3sz::Laughter;;/m/01j423::Yawn;;/m/01j4z9::Chainsaw;;/m/01jg02::Heart sounds, heartbeat;;/m/01jg1z::Heart murmur;;/m/01jnbd::Echo;;/m/01jt3m::Toilet flush;;/m/01jwx6::Vibration;;/m/01lsmm::Scissors;;/m/01lynh::Stairs;;/m/01m2v::Computer keyboard;;/m/01m4t::Printer;;/m/01rd7k::Turkey;;/m/01s0vc::Zipper (clothing);;/m/01sb50::Cellphone buzz, vibrating alert;;/m/01swy6::Yodeling;;/m/01v_m0::Sine wave;;/m/01w250::Whistling;;/m/01x3z::Clock;;/m/01xq0k1::Cattle, bovinae;;/m/01y3hg::Smoke detector, smoke alarm;;/m/01yg9g::Lawn mower;;/m/01yrx::Cat;;/m/01z47d::Busy signal;;/m/01z5f::Canidae, wild dogs, wolves;;/m/02021::Chipmunk;;/m/020bb7::Bird vocalization, bird call, bird song;;/m/0239kh::Cowbell;;/m/023pjk::Cutlery, silverware;;/m/023vsd::Sanding;;/m/02417f::Windscreen wiper, windshield wiper;;/m/0242l::Coin (dropping);;/m/024dl::Cash register;;/m/025_jnm::Finger snapping;;/m/025rv6n::Fowl;;/m/025wky1::Air conditioning;;/m/0261r1::Babbling;;/m/0269r2s::Chain;;/m/026fgl::Wind chime;;/m/027m70_::Jingle bell;;/m/0284vy3::Train horn;;/m/028ght::Applause;;/m/028v0c::Silence;;/m/02_41::Fire;;/m/02_nn::Fart;;/m/02bk07::Chant;;/m/02bm9n::Ratchet, pawl;;/m/02c8p::Telephone dialing, DTMF;;/m/02dgv::Door;;/m/02f9f_::Shower;;/m/02fs_r::Beep, bleep;;/m/02fxyj::Humming;;/m/02g901::Electric shaver, electric razor;;/m/02jz0l::Water tap, faucet;;/m/02l6bg::Propeller, airscrew;;/m/02ll1_::Lock;;/m/02mfyn::Car alarm;;/m/02mk9::Engine;;/m/02p01q::Filing (rasp);;/m/02p3nc::Hiccup;;/m/02pjr4::Blender, food processor;;/m/02qldy::Narration, monologue;;/m/02rhddq::Reversing beeps;;/m/02rlv9::Motorboat, speedboat;;/m/02rr_::Effects unit;;/m/02rtxlg::Whispering;;/m/02x984l::Mechanical fan;;/m/02y_763::Sliding door;;/m/02yds9::Purr;;/m/02z32qm::Fusillade;;/m/02zsn::Female speech, woman speaking;;/m/030rvx::Buzzer;;/m/0316dw::Typing;;/m/032n05::Whale vocalization;;/m/032s66::Gunshot, gunfire;;/m/034srq::Waves, surf;;/m/0395lw::Bell;;/m/039jq::Glass;;/m/03cczk::Chewing, mastication;;/m/03cl9h::Ice cream truck, ice cream van;;/m/03dnzn::Bathtub (filling or washing);;/m/03fwl::Goat;;/m/03j1ly::Emergency vehicle;;/m/03k3r::Horse;;/m/03kmc9::Siren;;/m/03l9g::Hammer;;/m/03m9d0z::Wind;;/m/03p19w::Jackhammer;;/m/03q5_w::Burping, eructation;;/m/03qc9zr::Screaming;;/m/03qtwd::Crowd;;/m/03v3yw::Keys jangling;;/m/03vt0::Insect;;/m/03w41f::Church bell;;/m/03wvsk::Hair dryer;;/m/03wwcy::Doorbell;;/m/040b_t::Refrigerator;;/m/04179zz::Duck call (hunting tool);;/m/04229::Jet engine;;/m/0463cq4::Crying, sobbing;;/m/046dlr::Alarm clock;;/m/04_sv::Motorcycle;;/m/04brg2::Dishes, pots, and pans;;/m/04ctx::Knife;;/m/04cvmfc::Roar;;/m/04fgwm::Electric toothbrush;;/m/04fq5q::Foghorn;;/m/04gxbd::Power windows, electric windows;;/m/04gy_2::Battle cry;;/m/04k94::Liquid;;/m/04qvtq::Police car (siren);;/m/04rlf::Music;;/m/04rmv::Mouse;;/m/04s8yn::Crow;;/m/04v5dt::Booing;;/m/04zjc::Machine gun;;/m/04zmvq::Train whistle;;/m/053hz1::Cheering;;/m/056ks2::Rowboat, canoe, kayak;;/m/056r_1::Keypress tone;;/m/05_wcq::Bird flight, flapping wings;;/m/05kq4::Ocean;;/m/05mxj0q::Packing tape, duct tape;;/m/05rj2::Shuffling cards;;/m/05tny_::Bark;;/m/05x_td::Air horn, truck horn;;/m/05zc1::Pulleys;;/m/05zppz::Male speech, man speaking;;/m/0641k::Paper rustling;;/m/0642b4::Cupboard open or close;;/m/068hy::Domestic animals, pets;;/m/068zj::Pig;;/m/06_fw::Skateboard;;/m/06_y0by::Environmental noise;;/m/06bxc::Rapping;;/m/06bz3::Radio;;/m/06cyt0::Mechanical bell;;/m/06d_3::Rail transport;;/m/06h7j::Run;;/m/06hck5::Steam whistle;;/m/06hps::Rodents, rats, mice;;/m/06mb1::Rain;;/m/06q74::Ship;;/m/06wzb::Steam;;/m/06xkwv::Mains hum;;/m/073cg4::Cap gun;;/m/078jl::Snake;;/m/0790c::Sonar;;/m/079vc8::Wolf-whistling;;/m/07bgp::Sheep;;/m/07bjf::Single-lens reflex camera;;/m/07bm98::Sound reproduction;;/m/07c52::Television;;/m/07cx4::Telephone;;/m/07hvw1::Unmodified field recording;;/m/07jdr::Train;;/m/07k1x::Tools;;/m/07m2kt::Chorus effect;;/m/07mzm6::Wheeze;;/m/07n_g::Tuning fork;;/m/07p6fty::Shout;;/m/07p6mqd::Slosh;;/m/07p78v5::Zing;;/m/07p7b8y::Fill (with liquid);;/m/07p9k1k::Sizzle;;/m/07p_0gm::Throbbing;;/m/07pb8fc::Idling;;/m/07pbtc8::Walk, footsteps;;/m/07pc8l3::Snap;;/m/07pc8lb::Breaking;;/m/07pczhz::Chop;;/m/07pdhp0::Biting;;/m/07pdjhy::Rub;;/m/07pggtn::Chirp, tweet;;/m/07phhsh::Rumble;;/m/07phxs1::Ding;;/m/07pjjrj::Smash, crash;;/m/07pjwq1::Buzz;;/m/07pk7mg::Wobble;;/m/07pl1bw::Splinter;;/m/07plct2::Crushing;;/m/07plz5l::Sigh;;/m/07pn_8q::Chopping (food);;/m/07pp8cl::Telephone bell ringing;;/m/07pp_mv::Alarm;;/m/07ppn3j::Sniff;;/m/07pqc89::Trickle, dribble;;/m/07pqmly::Slurp, drinking straw;;/m/07pqn27::Bouncing;;/m/07prgkl::Pour;;/m/07pt6mm::Grind;;/m/07pt_g0::Pulse;;/m/07ptfmf::Stir;;/m/07ptzwd::Pump (liquid);;/m/07pws3f::Bang;;/m/07pxg6y::Eruption;;/m/07pyf11::Flap;;/m/07pyy8b::Pant;;/m/07pzfmf::Crackle;;/m/07q0h5t::Bleat;;/m/07q0yl5::Snort;;/m/07q2z82::Accelerating, revving, vroom;;/m/07q34h3::Puff;;/m/07q4ntr::Bellow;;/m/07q5rw0::Neigh, whinny;;/m/07q6cd_::Squeak;;/m/07q7njn::Glass chink, clink;;/m/07q8f3b::Howl (wind);;/m/07q8k13::Screech;;/m/07qb_dv::Scratch;;/m/07qc9xj::Clicking;;/m/07qcpgn::Tap;;/m/07qcx4z::Tearing;;/m/07qdb04::Quack;;/m/07qf0zm::Howl;;/m/07qfgpx::Jingle, tinkle;;/m/07qfr4h::Hubbub, speech noise, speech babble;;/m/07qh7jl::Creak;;/m/07qjznl::Tick-tock;;/m/07qjznt::Tick;;/m/07qlf79::Spray;;/m/07qlwh6::Squish;;/m/07qmpdm::Clatter;;/m/07qn4z3::Rattle;;/m/07qn5dc::Crowing, cock-a-doodle-doo;;/m/07qnq_y::Thump, thud;;/m/07qqyl4::Boom;;/m/07qrkrw::Meow;;/m/07qs1cx::Crack;;/m/07qsvvw::Burst, pop;;/m/07qv4k0::Scrape;;/m/07qv_x_::Shuffle;;/m/07qw_06::Wail, moan;;/m/07qwdck::Ping;;/m/07qwf61::Honk;;/m/07qwyj0::Rustle;;/m/07qyrcz::Plop;;/m/07qz6j3::Whimper;;/m/07r04::Truck;;/m/07r10fb::Raindrop;;/m/07r4gkf::Patter;;/m/07r4k75::Grunt;;/m/07r4wb8::Knock;;/m/07r5c2p::Caw;;/m/07r5v4s::Drip;;/m/07r660_::Giggle;;/m/07r67yg::Ding-dong;;/m/07r81j2::Caterwaul;;/m/07r_25d::Coo;;/m/07r_80w::Hoot;;/m/07r_k2n::Yip;;/m/07rbp7_::Whip;;/m/07rc7d9::Bow-wow;;/m/07rcgpl::Hum;;/m/07rdhzs::Whack, thwack;;/m/07rgkc5::Static;;/m/07rgt08::Chuckle, chortle;;/m/07rjwbb::Hiss;;/m/07rjzl8::Slam;;/m/07rn7sz::Glass shatter;;/m/07rpkh9::Moo;;/m/07rqsjt::Whoosh, swoosh, swish;;/m/07rrh0c::Thunk;;/m/07rrlb6::Splash, splatter;;/m/07rv4dm::Clang;;/m/07rv9rh::Clip-clop;;/m/07rwj3x::Whoop;;/m/07rwm0c::Clickety-clack;;/m/07ryjzk::Slap, smack;;/m/07s02z0::Squeal;;/m/07s04w4::Snicker;;/m/07s0dtb::Gasp;;/m/07s12q4::Crunch;;/m/07s13rg::Sweeping;;/m/07s2xch::Groan;;/m/07s34ls::Whir;;/m/07s8j8t::Roll;;/m/07sk0jz::Stomp, stamp;;/m/07sq110::Belly laugh;;/m/07sr1lc::Yell;;/m/07st88b::Croak;;/m/07st89h::Cluck;;/m/07svc2k::Gobble;;/m/07swgks::Gurgling, bubbling;;/m/07sx8x_::Squawk;;/m/07szfh9::Cacophony;;/m/07yv9::Vehicle;;/m/081rb::Writing;;/m/0838f::Water;;/m/083vt::Wood;;/m/08dckq::Carbon monoxide detector, CO detector;;/m/08j51y::Dental drill, dentist's drill;;/m/0912c9::Vehicle horn, car horn, honking, toot;;/m/0939n_::Gargling;;/m/093_4n::Background noise;;/m/096m7z::Noise;;/m/098_xr::Error signal;;/m/09b5t::Chicken, rooster;;/m/09ct_::Helicopter;;/m/09d1b1::Tape hiss;;/m/09d5_::Owl;;/m/09ddx::Ducks, geese, waterfowl;;/m/09f96::Mosquito;;/m/09hlz4::Respiratory sounds;;/m/09l8g::Human voice;;/m/09ld4::Frog;;/m/09x0r::Speech;;/m/09xqv::Cricket;;/m/0_1c::Artillery fire;;/m/0_ksk::Power tool;;/m/0b_fwt::Electronic tuner;;/m/0bcdqg::Ringing tone, ringback tone;;/m/0bpl036::Human locomotion;;/m/0brhx::Speech synthesizer;;/m/0bt9lr::Dog;;/m/0btp2::Traffic noise, roadway noise;;/m/0bzvm2::Video game sound;;/m/0c1dj::Sound effect;;/m/0c1tlg::Electric rotor drone, quadcopter;;/m/0c2wf::Typewriter;;/m/0c3f7m::Fire alarm;;/m/0cdnk::Roaring cats (lions, tigers);;/m/0ch8v::Livestock, farm animals, working animals;;/m/0chx_::White noise, pink noise;;/m/0cmf2::Fixed-wing aircraft, airplane;;/m/0d31p::Vacuum cleaner;;/m/0d4wf::Kitchen and dining room sounds;;/m/0dgbq::Civil defense siren;;/m/0dgw9r::Human sounds;;/m/0dl83::Arrow;;/m/0dl9sf8::Throat clearing;;/m/0dv3j::Boiling;;/m/0dv5r::Camera;;/m/0dxrf::Frying (food);;/m/0f8s22::Chime;;/m/0ffhf::Donkey, ass;;/m/0fjy1::Wildfire;;/m/0fqfqc::Drawer open or close;;/m/0fw86::Tap dance;;/m/0fx9l::Microwave oven;;/m/0g12c5::Distortion;;/m/0g6b5::Fireworks;;/m/0ghcn6::Growling;;/m/0gvgw0::Air brake;;/m/0gy1t2s::Bicycle bell;;/m/0h0rv::Pigeon, dove;;/m/0h2mp::Fly, housefly;;/m/0h9mv::Tire squeal, skidding;;/m/0hdsk::Chirp tone;;/m/0hg7b::Microphone;;/m/0hgq8df::Crockery breaking and smashing;;/m/0hsrw::Sailboat, sailing ship;;/m/0j2kx::Waterfall;;/m/0j6m2::Stream, river;;/m/0jb2l::Thunderstorm;;/m/0jbk::Animal;;/m/0k4j::Car;;/m/0k5j::Aircraft;;/m/0k65p::Hands;;/m/0l14jd::Choir;;/m/0l156k::Whistle;;/m/0l15bq::Clapping;;/m/0l7xg::Gears;;/m/0llzx::Sewing machine;;/m/0ltv::Race car, auto racing;;/m/0lyf6::Breathing;;/m/0md09::Power saw, circular saw, table saw;;/m/0ngt1::Thunder;;/m/0ytgt::Child speech, kid speaking;;/m/0zmy2j9::Velcro, hook and loop fastener;;/t/dd00001::Baby laughter;;/t/dd00002::Baby cry, infant cry;;/t/dd00003::Male singing;;/t/dd00004::Female singing;;/t/dd00005::Child singing;;/t/dd00006::Synthetic singing;;/t/dd00012::Human group actions;;/t/dd00013::Children playing;;/t/dd00018::Oink;;/t/dd00038::Rain on surface;;/t/dd00048::Train wheels squealing;;/t/dd00061::Non-motorized land vehicle;;/t/dd00065::Light engine (high frequency);;/t/dd00066::Medium engine (mid frequency);;/t/dd00067::Heavy engine (low frequency);;/t/dd00077::Mechanisms;;/t/dd00088::Gush;;/t/dd00091::Sound equipment;;/t/dd00092::Wind noise (microphone);;/t/dd00098::Source-ambiguous sounds;;/t/dd00099::Generic impact sounds;;/t/dd00108::Clunk;;/t/dd00109::Surface contact;;/t/dd00110::Deformable shell;;/t/dd00112::Crumpling, crinkling;;/t/dd00118::Fizz;;/t/dd00121::Boing;;/t/dd00122::Other sourceless;;/t/dd00123::Channel, environment and background;;/t/dd00125::Inside, small room;;/t/dd00126::Inside, large room or hall;;/t/dd00127::Inside, public space;;/t/dd00128::Outside, urban or manmade;;/t/dd00129::Outside, rural or natural;;/t/dd00130::Engine starting;;/t/dd00133::Specific impact sounds;;/t/dd00134::Car passing by;;/t/dd00135::Children shouting;;/t/dd00136::Whimper (dog);;/t/dd00138::Brief tone;;/t/dd00139::Snort (horse);;/t/dd00141::Pant (dog);;/t/dd00142::Audio logo;;/t/dd00143::Unknown sound;;/t/dd00144::Alert;;/t/dd00147::Dong, bong++labels_weak++/m/09x0r::Speech;;/m/05zppz::Male speech, man speaking;;/m/02zsn::Female speech, woman speaking;;/m/0ytgt::Child speech, kid speaking;;/m/01h8n0::Conversation;;/m/02qldy::Narration, monologue;;/m/0261r1::Babbling;;/m/0brhx::Speech synthesizer;;/m/07p6fty::Shout;;/m/07q4ntr::Bellow;;/m/07rwj3x::Whoop;;/m/07sr1lc::Yell;;/m/04gy_2::Battle cry;;/t/dd00135::Children shouting;;/m/03qc9zr::Screaming;;/m/02rtxlg::Whispering;;/m/01j3sz::Laughter;;/t/dd00001::Baby laughter;;/m/07r660_::Giggle;;/m/07s04w4::Snicker;;/m/07sq110::Belly laugh;;/m/07rgt08::Chuckle, chortle;;/m/0463cq4::Crying, sobbing;;/t/dd00002::Baby cry, infant cry;;/m/07qz6j3::Whimper;;/m/07qw_06::Wail, moan;;/m/07plz5l::Sigh;;/m/015lz1::Singing;;/m/0l14jd::Choir;;/m/01swy6::Yodeling;;/m/02bk07::Chant;;/m/01c194::Mantra;;/t/dd00003::Male singing;;/t/dd00004::Female singing;;/t/dd00005::Child singing;;/t/dd00006::Synthetic singing;;/m/06bxc::Rapping;;/m/02fxyj::Humming;;/m/07s2xch::Groan;;/m/07r4k75::Grunt;;/m/01w250::Whistling;;/m/0lyf6::Breathing;;/m/07mzm6::Wheeze;;/m/01d3sd::Snoring;;/m/07s0dtb::Gasp;;/m/07pyy8b::Pant;;/m/07q0yl5::Snort;;/m/01b_21::Cough;;/m/0dl9sf8::Throat clearing;;/m/01hsr_::Sneeze;;/m/07ppn3j::Sniff;;/m/06h7j::Run;;/m/07qv_x_::Shuffle;;/m/07pbtc8::Walk, footsteps;;/m/03cczk::Chewing, mastication;;/m/07pdhp0::Biting;;/m/0939n_::Gargling;;/m/01g90h::Stomach rumble;;/m/03q5_w::Burping, eructation;;/m/02p3nc::Hiccup;;/m/02_nn::Fart;;/m/0k65p::Hands;;/m/025_jnm::Finger snapping;;/m/0l15bq::Clapping;;/m/01jg02::Heart sounds, heartbeat;;/m/01jg1z::Heart murmur;;/m/053hz1::Cheering;;/m/028ght::Applause;;/m/07rkbfh::Chatter;;/m/03qtwd::Crowd;;/m/07qfr4h::Hubbub, speech noise, speech babble;;/t/dd00013::Children playing;;/m/0jbk::Animal;;/m/068hy::Domestic animals, pets;;/m/0bt9lr::Dog;;/m/05tny_::Bark;;/m/07r_k2n::Yip;;/m/07qf0zm::Howl;;/m/07rc7d9::Bow-wow;;/m/0ghcn6::Growling;;/t/dd00136::Whimper (dog);;/m/01yrx::Cat;;/m/02yds9::Purr;;/m/07qrkrw::Meow;;/m/07rjwbb::Hiss;;/m/07r81j2::Caterwaul;;/m/0ch8v::Livestock, farm animals, working animals;;/m/03k3r::Horse;;/m/07rv9rh::Clip-clop;;/m/07q5rw0::Neigh, whinny;;/m/01xq0k1::Cattle, bovinae;;/m/07rpkh9::Moo;;/m/0239kh::Cowbell;;/m/068zj::Pig;;/t/dd00018::Oink;;/m/03fwl::Goat;;/m/07q0h5t::Bleat;;/m/07bgp::Sheep;;/m/025rv6n::Fowl;;/m/09b5t::Chicken, rooster;;/m/07st89h::Cluck;;/m/07qn5dc::Crowing, cock-a-doodle-doo;;/m/01rd7k::Turkey;;/m/07svc2k::Gobble;;/m/09ddx::Duck;;/m/07qdb04::Quack;;/m/0dbvp::Goose;;/m/07qwf61::Honk;;/m/01280g::Wild animals;;/m/0cdnk::Roaring cats (lions, tigers);;/m/04cvmfc::Roar;;/m/015p6::Bird;;/m/020bb7::Bird vocalization, bird call, bird song;;/m/07pggtn::Chirp, tweet;;/m/07sx8x_::Squawk;;/m/0h0rv::Pigeon, dove;;/m/07r_25d::Coo;;/m/04s8yn::Crow;;/m/07r5c2p::Caw;;/m/09d5_::Owl;;/m/07r_80w::Hoot;;/m/05_wcq::Bird flight, flapping wings;;/m/01z5f::Canidae, dogs, wolves;;/m/06hps::Rodents, rats, mice;;/m/04rmv::Mouse;;/m/07r4gkf::Patter;;/m/03vt0::Insect;;/m/09xqv::Cricket;;/m/09f96::Mosquito;;/m/0h2mp::Fly, housefly;;/m/07pjwq1::Buzz;;/m/01h3n::Bee, wasp, etc.;;/m/09ld4::Frog;;/m/07st88b::Croak;;/m/078jl::Snake;;/m/07qn4z3::Rattle;;/m/032n05::Whale vocalization;;/m/04rlf::Music;;/m/04szw::Musical instrument;;/m/0fx80y::Plucked string instrument;;/m/0342h::Guitar;;/m/02sgy::Electric guitar;;/m/018vs::Bass guitar;;/m/042v_gx::Acoustic guitar;;/m/06w87::Steel guitar, slide guitar;;/m/01glhc::Tapping (guitar technique);;/m/07s0s5r::Strum;;/m/018j2::Banjo;;/m/0jtg0::Sitar;;/m/04rzd::Mandolin;;/m/01bns_::Zither;;/m/07xzm::Ukulele;;/m/05148p4::Keyboard (musical);;/m/05r5c::Piano;;/m/01s0ps::Electric piano;;/m/013y1f::Organ;;/m/03xq_f::Electronic organ;;/m/03gvt::Hammond organ;;/m/0l14qv::Synthesizer;;/m/01v1d8::Sampler;;/m/03q5t::Harpsichord;;/m/0l14md::Percussion;;/m/02hnl::Drum kit;;/m/0cfdd::Drum machine;;/m/026t6::Drum;;/m/06rvn::Snare drum;;/m/03t3fj::Rimshot;;/m/02k_mr::Drum roll;;/m/0bm02::Bass drum;;/m/011k_j::Timpani;;/m/01p970::Tabla;;/m/01qbl::Cymbal;;/m/03qtq::Hi-hat;;/m/01sm1g::Wood block;;/m/07brj::Tambourine;;/m/05r5wn::Rattle (instrument);;/m/0xzly::Maraca;;/m/0mbct::Gong;;/m/016622::Tubular bells;;/m/0j45pbj::Mallet percussion;;/m/0dwsp::Marimba, xylophone;;/m/0dwtp::Glockenspiel;;/m/0dwt5::Vibraphone;;/m/0l156b::Steelpan;;/m/05pd6::Orchestra;;/m/01kcd::Brass instrument;;/m/0319l::French horn;;/m/07gql::Trumpet;;/m/07c6l::Trombone;;/m/0l14_3::Bowed string instrument;;/m/02qmj0d::String section;;/m/07y_7::Violin, fiddle;;/m/0d8_n::Pizzicato;;/m/01xqw::Cello;;/m/02fsn::Double bass;;/m/085jw::Wind instrument, woodwind instrument;;/m/0l14j_::Flute;;/m/06ncr::Saxophone;;/m/01wy6::Clarinet;;/m/03m5k::Harp;;/m/0395lw::Bell;;/m/03w41f::Church bell;;/m/027m70_::Jingle bell;;/m/0gy1t2s::Bicycle bell;;/m/07n_g::Tuning fork;;/m/0f8s22::Chime;;/m/026fgl::Wind chime;;/m/0150b9::Change ringing (campanology);;/m/03qjg::Harmonica;;/m/0mkg::Accordion;;/m/0192l::Bagpipes;;/m/02bxd::Didgeridoo;;/m/0l14l2::Shofar;;/m/07kc_::Theremin;;/m/0l14t7::Singing bowl;;/m/01hgjl::Scratching (performance technique);;/m/064t9::Pop music;;/m/0glt670::Hip hop music;;/m/02cz_7::Beatboxing;;/m/06by7::Rock music;;/m/03lty::Heavy metal;;/m/05r6t::Punk rock;;/m/0dls3::Grunge;;/m/0dl5d::Progressive rock;;/m/07sbbz2::Rock and roll;;/m/05w3f::Psychedelic rock;;/m/06j6l::Rhythm and blues;;/m/0gywn::Soul music;;/m/06cqb::Reggae;;/m/01lyv::Country;;/m/015y_n::Swing music;;/m/0gg8l::Bluegrass;;/m/02x8m::Funk;;/m/02w4v::Folk music;;/m/06j64v::Middle Eastern music;;/m/03_d0::Jazz;;/m/026z9::Disco;;/m/0ggq0m::Classical music;;/m/05lls::Opera;;/m/02lkt::Electronic music;;/m/03mb9::House music;;/m/07gxw::Techno;;/m/07s72n::Dubstep;;/m/0283d::Drum and bass;;/m/0m0jc::Electronica;;/m/08cyft::Electronic dance music;;/m/0fd3y::Ambient music;;/m/07lnk::Trance music;;/m/0g293::Music of Latin America;;/m/0ln16::Salsa music;;/m/0326g::Flamenco;;/m/0155w::Blues;;/m/05fw6t::Music for children;;/m/02v2lh::New-age music;;/m/0y4f8::Vocal music;;/m/0z9c::A capella;;/m/0164x2::Music of Africa;;/m/0145m::Afrobeat;;/m/02mscn::Christian music;;/m/016cjb::Gospel music;;/m/028sqc::Music of Asia;;/m/015vgc::Carnatic music;;/m/0dq0md::Music of Bollywood;;/m/06rqw::Ska;;/m/02p0sh1::Traditional music;;/m/05rwpb::Independent music;;/m/074ft::Song;;/m/025td0t::Background music;;/m/02cjck::Theme music;;/m/03r5q_::Jingle (music);;/m/0l14gg::Soundtrack music;;/m/07pkxdp::Lullaby;;/m/01z7dr::Video game music;;/m/0140xf::Christmas music;;/m/0ggx5q::Dance music;;/m/04wptg::Wedding music;;/t/dd00031::Happy music;;/t/dd00032::Funny music;;/t/dd00033::Sad music;;/t/dd00034::Tender music;;/t/dd00035::Exciting music;;/t/dd00036::Angry music;;/t/dd00037::Scary music;;/m/03m9d0z::Wind;;/m/09t49::Rustling leaves;;/t/dd00092::Wind noise (microphone);;/m/0jb2l::Thunderstorm;;/m/0ngt1::Thunder;;/m/0838f::Water;;/m/06mb1::Rain;;/m/07r10fb::Raindrop;;/t/dd00038::Rain on surface;;/m/0j6m2::Stream;;/m/0j2kx::Waterfall;;/m/05kq4::Ocean;;/m/034srq::Waves, surf;;/m/06wzb::Steam;;/m/07swgks::Gurgling;;/m/02_41::Fire;;/m/07pzfmf::Crackle;;/m/07yv9::Vehicle;;/m/019jd::Boat, Water vehicle;;/m/0hsrw::Sailboat, sailing ship;;/m/056ks2::Rowboat, canoe, kayak;;/m/02rlv9::Motorboat, speedboat;;/m/06q74::Ship;;/m/012f08::Motor vehicle (road);;/m/0k4j::Car;;/m/0912c9::Vehicle horn, car horn, honking;;/m/07qv_d5::Toot;;/m/02mfyn::Car alarm;;/m/04gxbd::Power windows, electric windows;;/m/07rknqz::Skidding;;/m/0h9mv::Tire squeal;;/t/dd00134::Car passing by;;/m/0ltv::Race car, auto racing;;/m/07r04::Truck;;/m/0gvgw0::Air brake;;/m/05x_td::Air horn, truck horn;;/m/02rhddq::Reversing beeps;;/m/03cl9h::Ice cream truck, ice cream van;;/m/01bjv::Bus;;/m/03j1ly::Emergency vehicle;;/m/04qvtq::Police car (siren);;/m/012n7d::Ambulance (siren);;/m/012ndj::Fire engine, fire truck (siren);;/m/04_sv::Motorcycle;;/m/0btp2::Traffic noise, roadway noise;;/m/06d_3::Rail transport;;/m/07jdr::Train;;/m/04zmvq::Train whistle;;/m/0284vy3::Train horn;;/m/01g50p::Railroad car, train wagon;;/t/dd00048::Train wheels squealing;;/m/0195fx::Subway, metro, underground;;/m/0k5j::Aircraft;;/m/014yck::Aircraft engine;;/m/04229::Jet engine;;/m/02l6bg::Propeller, airscrew;;/m/09ct_::Helicopter;;/m/0cmf2::Fixed-wing aircraft, airplane;;/m/0199g::Bicycle;;/m/06_fw::Skateboard;;/m/02mk9::Engine;;/t/dd00065::Light engine (high frequency);;/m/08j51y::Dental drill, dentist's drill;;/m/01yg9g::Lawn mower;;/m/01j4z9::Chainsaw;;/t/dd00066::Medium engine (mid frequency);;/t/dd00067::Heavy engine (low frequency);;/m/01h82_::Engine knocking;;/t/dd00130::Engine starting;;/m/07pb8fc::Idling;;/m/07q2z82::Accelerating, revving, vroom;;/m/02dgv::Door;;/m/03wwcy::Doorbell;;/m/07r67yg::Ding-dong;;/m/02y_763::Sliding door;;/m/07rjzl8::Slam;;/m/07r4wb8::Knock;;/m/07qcpgn::Tap;;/m/07q6cd_::Squeak;;/m/0642b4::Cupboard open or close;;/m/0fqfqc::Drawer open or close;;/m/04brg2::Dishes, pots, and pans;;/m/023pjk::Cutlery, silverware;;/m/07pn_8q::Chopping (food);;/m/0dxrf::Frying (food);;/m/0fx9l::Microwave oven;;/m/02pjr4::Blender;;/m/02jz0l::Water tap, faucet;;/m/0130jx::Sink (filling or washing);;/m/03dnzn::Bathtub (filling or washing);;/m/03wvsk::Hair dryer;;/m/01jt3m::Toilet flush;;/m/012xff::Toothbrush;;/m/04fgwm::Electric toothbrush;;/m/0d31p::Vacuum cleaner;;/m/01s0vc::Zipper (clothing);;/m/03v3yw::Keys jangling;;/m/0242l::Coin (dropping);;/m/01lsmm::Scissors;;/m/02g901::Electric shaver, electric razor;;/m/05rj2::Shuffling cards;;/m/0316dw::Typing;;/m/0c2wf::Typewriter;;/m/01m2v::Computer keyboard;;/m/081rb::Writing;;/m/07pp_mv::Alarm;;/m/07cx4::Telephone;;/m/07pp8cl::Telephone bell ringing;;/m/01hnzm::Ringtone;;/m/02c8p::Telephone dialing, DTMF;;/m/015jpf::Dial tone;;/m/01z47d::Busy signal;;/m/046dlr::Alarm clock;;/m/03kmc9::Siren;;/m/0dgbq::Civil defense siren;;/m/030rvx::Buzzer;;/m/01y3hg::Smoke detector, smoke alarm;;/m/0c3f7m::Fire alarm;;/m/04fq5q::Foghorn;;/m/0l156k::Whistle;;/m/06hck5::Steam whistle;;/t/dd00077::Mechanisms;;/m/02bm9n::Ratchet, pawl;;/m/01x3z::Clock;;/m/07qjznt::Tick;;/m/07qjznl::Tick-tock;;/m/0l7xg::Gears;;/m/05zc1::Pulleys;;/m/0llzx::Sewing machine;;/m/02x984l::Mechanical fan;;/m/025wky1::Air conditioning;;/m/024dl::Cash register;;/m/01m4t::Printer;;/m/0dv5r::Camera;;/m/07bjf::Single-lens reflex camera;;/m/07k1x::Tools;;/m/03l9g::Hammer;;/m/03p19w::Jackhammer;;/m/01b82r::Sawing;;/m/02p01q::Filing (rasp);;/m/023vsd::Sanding;;/m/0_ksk::Power tool;;/m/01d380::Drill;;/m/014zdl::Explosion;;/m/032s66::Gunshot, gunfire;;/m/04zjc::Machine gun;;/m/02z32qm::Fusillade;;/m/0_1c::Artillery fire;;/m/073cg4::Cap gun;;/m/0g6b5::Fireworks;;/g/122z_qxw::Firecracker;;/m/07qsvvw::Burst, pop;;/m/07pxg6y::Eruption;;/m/07qqyl4::Boom;;/m/083vt::Wood;;/m/07pczhz::Chop;;/m/07pl1bw::Splinter;;/m/07qs1cx::Crack;;/m/039jq::Glass;;/m/07q7njn::Chink, clink;;/m/07rn7sz::Shatter;;/m/04k94::Liquid;;/m/07rrlb6::Splash, splatter;;/m/07p6mqd::Slosh;;/m/07qlwh6::Squish;;/m/07r5v4s::Drip;;/m/07prgkl::Pour;;/m/07pqc89::Trickle, dribble;;/t/dd00088::Gush;;/m/07p7b8y::Fill (with liquid);;/m/07qlf79::Spray;;/m/07ptzwd::Pump (liquid);;/m/07ptfmf::Stir;;/m/0dv3j::Boiling;;/m/0790c::Sonar;;/m/0dl83::Arrow;;/m/07rqsjt::Whoosh, swoosh, swish;;/m/07qnq_y::Thump, thud;;/m/07rrh0c::Thunk;;/m/0b_fwt::Electronic tuner;;/m/02rr_::Effects unit;;/m/07m2kt::Chorus effect;;/m/018w8::Basketball bounce;;/m/07pws3f::Bang;;/m/07ryjzk::Slap, smack;;/m/07rdhzs::Whack, thwack;;/m/07pjjrj::Smash, crash;;/m/07pc8lb::Breaking;;/m/07pqn27::Bouncing;;/m/07rbp7_::Whip;;/m/07pyf11::Flap;;/m/07qb_dv::Scratch;;/m/07qv4k0::Scrape;;/m/07pdjhy::Rub;;/m/07s8j8t::Roll;;/m/07plct2::Crushing;;/t/dd00112::Crumpling, crinkling;;/m/07qcx4z::Tearing;;/m/02fs_r::Beep, bleep;;/m/07qwdck::Ping;;/m/07phxs1::Ding;;/m/07rv4dm::Clang;;/m/07s02z0::Squeal;;/m/07qh7jl::Creak;;/m/07qwyj0::Rustle;;/m/07s34ls::Whir;;/m/07qmpdm::Clatter;;/m/07p9k1k::Sizzle;;/m/07qc9xj::Clicking;;/m/07rwm0c::Clickety-clack;;/m/07phhsh::Rumble;;/m/07qyrcz::Plop;;/m/07qfgpx::Jingle, tinkle;;/m/07rcgpl::Hum;;/m/07p78v5::Zing;;/t/dd00121::Boing;;/m/07s12q4::Crunch;;/m/028v0c::Silence;;/m/01v_m0::Sine wave;;/m/0b9m1::Harmonic;;/m/0hdsk::Chirp tone;;/m/0c1dj::Sound effect;;/m/07pt_g0::Pulse;;/t/dd00125::Inside, small room;;/t/dd00126::Inside, large room or hall;;/t/dd00127::Inside, public space;;/t/dd00128::Outside, urban or manmade;;/t/dd00129::Outside, rural or natural;;/m/01b9nn::Reverberation;;/m/01jnbd::Echo;;/m/096m7z::Noise;;/m/06_y0by::Environmental noise;;/m/07rgkc5::Static;;/m/06xkwv::Mains hum;;/m/0g12c5::Distortion;;/m/08p9q4::Sidetone;;/m/07szfh9::Cacophony;;/m/0chx_::White noise;;/m/0cj0r::Pink noise;;/m/07p_0gm::Throbbing;;/m/01jwx6::Vibration;;/m/07c52::Television;;/m/06bz3::Radio;;/m/07hvw1::Field recording"

_HOMEPAGE = "https://research.google.com/audioset/download_strong.html"

_LICENSE = "https://creativecommons.org/licenses/by/4.0/"

label_map_strong_csv = os.path.join("metadata", "class_labels_indices_strong.csv")
label_map_csv = os.path.join("metadata", "class_labels_indices.csv")

# TODO: set base path to the location where you store AudioSet
base_path = "/share/hel/datasets/audioset"

# TODO: AudioSet Audio Data Files, in our case they are in mp3 format
_AUDIO_FILES = {
    "balanced_train": os.path.join(base_path, "hdf5s", "balanced_train_segments_mp3.hdf"),
    "unbalanced_train": os.path.join(base_path, "hdf5s", "unbalanced_train_segments_mp3.hdf"),
    "eval": os.path.join(base_path, "hdf5s", "eval_segments_mp3.hdf")
}

_METADATA = {
    "balanced_train": os.path.join("metadata", "audioset_train_strong.csv"),
    "unbalanced_train": os.path.join("metadata", "audioset_train_strong.csv"),
    "eval": os.path.join("metadata", "audioset_eval_strong.csv")
}


class AudiosetStrong(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features_dict = {
            'events': Sequence(feature={
                'event_label': Value(dtype='string', id=None),
                'event_label_id': Value(dtype='string', id=None),
                'offset': Value(dtype='float32', id=None),
                'onset': Value(dtype='float32', id=None)}, length=-1, id=None),
            'labels': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'label_ids': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'filepath': Value(dtype='string', id=None),
            'filename': Value(dtype='string', id=None)
        }

        features = datasets.Features(
            {
                **features_dict,
                "mp3_bytes": datasets.Value("binary")
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download. DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        split_csvs = dl_manager.download(_METADATA)
        hdf5s = dl_manager.download(_AUDIO_FILES)

        return [
            datasets.SplitGenerator(
                name=datasets.Split(key),
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": key,
                    "csv_path": split_csvs[key],
                    "hdf5_path": hdf5s[key],
                    "audio_path": os.path.join(base_path, 'audio'),
                    "label_map_csv": dl_manager.download(label_map_csv),
                    "label_map_strong_csv": dl_manager.download(label_map_strong_csv)
                },
            )
            for key in _METADATA
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, split, csv_path, hdf5_path, audio_path, label_map_csv, label_map_strong_csv):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # load mapping of label IDs to label names
        with open(label_map_strong_csv, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        label_id_map_strong = dict(lines)

        with open(label_map_csv, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)[1:]

        idx_id_map = dict([(int(r[0]), r[1]) for r in lines])
        idx_label_map = dict([(int(r[0]), r[2]) for r in lines])

        # load file with mp3s
        with h5py.File(hdf5_path, 'r') as f:
            mp3s = f['mp3']
            target = f['target']

            filenames = [n.decode('UTF-8').rsplit('.', 1)[0].split('/', 1)[-1] for n in f['audio_name']]
            name_to_idx = dict(zip(filenames, range(0, len(filenames))))

            df = pd.read_csv(csv_path, index_col=0)
            df['label_id'] = df['label']
            df['label'] = df['label'].map(label_id_map_strong)

            for idx in df.index.unique():
                yid = str(idx)
                yid = "Y" + yid.rsplit('_', 1)[0]
                if name_to_idx.get(yid) is None:
                    # print(f'{yid} is not in hdf5 file')
                    continue

                seq_list = df.loc[[idx]]  # get rows corresponding to the file

                event_list = []
                for index, row in seq_list.iterrows():
                    event = dict()
                    event["onset"] = float(row["start_time_seconds"])
                    event["offset"] = float(row["end_time_seconds"])
                    event["event_label"] = str(row["label"])
                    event["event_label_id"] = str(row["label_id"])
                    event_list.append(event)

                data_row = {
                    "label_ids": [idx_id_map[i] for i in np.where(target[name_to_idx[yid]])[0]],
                    "labels": [idx_label_map[i] for i in np.where(target[name_to_idx[yid]])[0]],
                    "filepath": os.path.join(audio_path, str(yid) + '.mp3'),
                    "filename": str(yid) + '.mp3',
                    "events": event_list
                }

                # load mp3 bytes file
                audio = mp3s[name_to_idx[yid]].tobytes()

                data_row = {**data_row, "mp3_bytes": audio}
                yield yid, data_row


if __name__ == '__main__':
    import datasets

    datasets.config.MAX_SHARD_SIZE = "2GB"
    datasets.logging.set_verbosity_info()

    datasets.config.IN_MEMORY_MAX_SIZE = 10 * 1024 * 1024 * 1024

    dataset = datasets.load_dataset("audioset_strong.py", num_proc=1, download_mode="force_redownload",
                                    trust_remote_code=True)

    # TODO: specify location you want to store AudioSet Strong Huggingface Dataset in
    dataset.save_to_disk(
        f"/share/hel/datasets/HF_datasets/local/audioset_strong_official",
        max_shard_size="2GB",
        num_proc=1,
    )
