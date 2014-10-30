#include <minerva.h>

using namespace std;
using namespace minerva;

float input_raw[] = {8.59232186e-01, -3.67248891e-01, -6.32162377e-01, -5.90879443e-01, 1.35450058e-01, 1.91089406e-01, 9.29029039e-01, 3.06354194e-01, 4.97813275e-01, 3.07139742e-01, 4.95429619e-01, 9.22613472e-01, -9.83223404e-01, -7.87111247e-01, -4.02592572e-01, 3.12822366e-01, 6.19625105e-01, 7.44351827e-01, 9.29295195e-01, 4.47370694e-01, 2.84950656e-01, 4.34907242e-01, -6.48019856e-02, -3.48830645e-01, -1.20710788e-01, 4.59378165e-01, 9.88029172e-01, 3.53747424e-01, 5.81645036e-01, -6.58171484e-01, -9.46301448e-01, 6.00740488e-01, 8.07445076e-01, -9.50647579e-01, -1.65053631e-02, 5.25103347e-02, 1.92732021e-01, -8.96084910e-01, 7.90179056e-01, 4.56532361e-01, 6.36700023e-01, 4.45505669e-04, 6.20378818e-01, -8.08062949e-01, -5.62099913e-01, -4.82561877e-01, -6.37884921e-02, -8.12535948e-02, 4.19019560e-01, -6.43893988e-01, 6.28997687e-02, -6.64515542e-01, 5.37627837e-01, 8.56341098e-01, 2.18987316e-01, -6.99633011e-01, -2.07465926e-02, -2.45310092e-01, 6.97202824e-01, 8.22194457e-01, -2.32302558e-01, -3.69008193e-01, 1.36788306e-01, -6.24363930e-01, -7.48316912e-01, 3.75191610e-01, 5.99213436e-01, 1.47073130e-01, 9.46459963e-01, 2.68108754e-01, 7.76843450e-01, -9.17048249e-03, -2.96766940e-01, 4.28460737e-01, 7.85823290e-03, -5.48724787e-01, -5.10051120e-01, 5.85601400e-01, -9.65517098e-03, 8.30187347e-01, 8.90743668e-01, 6.64644593e-02, -4.95014811e-01, 4.41724116e-01, -2.65122472e-01, -2.70311418e-03, -5.46849905e-01, -2.92868707e-01, 3.01703573e-01, -3.74134209e-01, 5.37470894e-01, 5.63674207e-01, 7.04818966e-01, 8.99811480e-01, -7.85354176e-01, 8.21450712e-01, -3.27889676e-01, 6.52760854e-01, 7.96201270e-01, -9.14569391e-01, -6.08410002e-01, -4.10997356e-01, 2.53999761e-01, -8.27553790e-01, -7.14109960e-01, 3.16530385e-02, 3.78682659e-01, 7.13251622e-01, 2.94723367e-01, 1.63237351e-01, 4.22231910e-01, -4.95166286e-01, 8.00319367e-01, -1.15412614e-01, -9.58958351e-01, 9.19322028e-01, 3.04450845e-01, 2.64125002e-02, 3.64712766e-01, -2.09192187e-02, 8.52980343e-01, 3.17595443e-02, -8.55680237e-01, 1.35016596e-01, 2.30486367e-01, 8.83092589e-01, -1.69273291e-01, -4.71120051e-01, -8.05213669e-01, -2.83115562e-02, -7.06742743e-02, -9.40481366e-01, 3.88554924e-01, 4.33894225e-01, 4.59622847e-01, -1.71297966e-01, -9.69802310e-01, 8.17950315e-01, 5.78757436e-01, -6.69601662e-01, -3.74428077e-01, 2.21890612e-01, -2.71019427e-01, -6.87922822e-01, -6.45392373e-01, 7.35779342e-01, -4.19810663e-01, 1.70359243e-01, -9.20102482e-02, -1.77643736e-01, 7.65268890e-01, 3.85416030e-01, -4.41453290e-01, -8.71119538e-01, -6.02752772e-01, 8.63365489e-01, 7.08827136e-01, 9.09469469e-01, -8.95493304e-01, 1.58943361e-01, -3.90074667e-02, -9.56582042e-01, -2.52759073e-01, -1.71816398e-01, 2.07814468e-01, 3.43497455e-01, 6.77731401e-01, 5.59052417e-01, -1.98597912e-01, 5.89058463e-01, 7.86248621e-01, -4.75020618e-01, 9.78394015e-01, 7.06614198e-01, 4.62954312e-01, -2.88868751e-01, 7.66578981e-01, 7.35918182e-01, 9.11532892e-01, -9.99785487e-01, -9.66917916e-01, -3.71859885e-01, 9.90632349e-01, -7.02311554e-01, -6.65245763e-01, 5.17144074e-01, -8.60940668e-01, 4.10946877e-01, -6.16694364e-02, -9.79623568e-01, 5.49647726e-01, 5.88402017e-01, -7.00861097e-01, -9.52592736e-01, 5.24127542e-01, -5.52659640e-01, -4.75651204e-01, -8.62609944e-02, -5.00146163e-01, 1.36567123e-01, 6.93885994e-01, -2.43800926e-01, -1.35069767e-01, 6.65238353e-01, -2.57736129e-01, -9.18893456e-01, 1.09342972e-01, -9.75075144e-02, 4.50601953e-01, -2.43098955e-01, 6.81324991e-01, -6.13706128e-02, 1.25286858e-01, 3.22398678e-01, -7.55162650e-02, 2.47273891e-01, -5.56238738e-01, 4.65726234e-01, -2.36635823e-01, -6.10330916e-01, -4.57674450e-01, -5.01549896e-01, -6.95721875e-01, 5.42747408e-01, -4.89176535e-01, -7.44913190e-01, 3.30334307e-01, -1.74390103e-01, 3.35535533e-01, 3.19627034e-01, -3.89244656e-01, -5.97551518e-01, -5.55945747e-01, -7.60058273e-01, -9.25709118e-01, -9.31736833e-01, -5.39006904e-01, -5.43292587e-01, 2.49821244e-01, 7.85122371e-01, 5.59456032e-01, 4.42902537e-01, -3.79115682e-01, -2.73833167e-01, -6.07836432e-01, 8.70983596e-01, 1.23468000e-01, 6.34583074e-01, -3.02172038e-01, 5.99428526e-01, -7.91791075e-01, 4.24240330e-01, 8.34896992e-01, 6.07170737e-01, -3.45773707e-01, -4.89785641e-01, -9.99565129e-03, -1.72778091e-01, -1.50125809e-01, -8.51243390e-01, 2.06781303e-01, 4.94399467e-01, 5.95152453e-01, -2.36998955e-01, 5.94316306e-01, -5.64052608e-02, 4.42798342e-01, -6.21574621e-01, -1.30808581e-01, 6.46936218e-01, 6.52545256e-01, -6.20949033e-01, -9.59795660e-01, 4.06982772e-01, -3.05459761e-01, -1.60992368e-01, 5.36177806e-01, 9.25756133e-01, 7.85130614e-01, -7.30115467e-01, 5.95609430e-01, 3.64181215e-01, 6.01057742e-02, 7.31963310e-01, 5.06496191e-01, -8.13594826e-01, -3.41121136e-01, -1.75274609e-01};

float weight_raw[] = {-2.99421235e+00, 5.85381379e-01, 1.09536925e+00, -8.02315431e-01, -6.21006855e-01, -1.47845127e-01, 4.86479403e-01, -2.17717723e+00, 2.99648504e+00, 4.39632527e-02, -4.15997727e-02, -1.87875147e+00, -1.22347191e+00, 1.79109036e+00, -1.23133024e+00, 1.18272956e+00, -1.36224463e+00, 2.47517071e+00, -1.20876460e+00, 2.33915863e+00, 2.68033376e+00, -1.98652306e+00, -8.44251566e-01, 1.09306382e+00, 2.52835182e+00, -2.13045394e+00, -2.40075369e+00, -5.30383341e-01, 2.85075222e+00, -2.75096075e+00, -1.87851423e+00, 6.52607928e-01, -2.47668771e+00, -1.10227108e+00, 6.60814659e-01, -1.79283172e+00, 1.15138630e+00, -1.53223817e+00, 1.08223018e+00, -8.43736265e-02, -1.43703113e+00, -1.30355808e+00, 2.43270972e+00, -1.34494388e+00, 2.04688826e+00, -1.82145375e+00, -1.01649942e+00, 2.67977931e+00, 3.64310972e-02, -5.79504850e-01, -2.83763077e+00, 7.30642137e-01, -9.15908959e-01, -1.33920463e+00, -2.61567136e+00, -9.93656887e-01, -2.59374035e+00, -2.97705451e+00, -2.75419318e+00, -9.06270806e-01, -2.44827413e+00, 1.21894359e-01, -1.70550112e+00, 2.94999927e+00, -1.39687703e+00, 1.08100832e+00, 1.57476715e-01, -3.60461582e-01, 9.69331474e-01, -2.19128895e+00, 1.72709403e+00, -2.17213379e+00, 7.97539786e-01, -1.71800785e+00, -2.85974811e+00, 7.61603840e-01, 8.52982359e-01, 8.52620021e-01, 2.18772540e+00, -9.76791579e-01, -5.34249680e-01, 6.61912103e-01, 2.64505914e+00, -2.08753638e+00, -1.89565196e+00, -2.07337027e+00, 1.27694596e+00, 2.56789897e+00, -4.63876920e-01, 2.71063582e-01, -1.06566621e+00, -2.03577190e+00, 1.21891215e+00, 1.38149044e+00, 2.76242723e+00, -7.77056575e-01, 1.30019035e+00, 2.91636731e+00, -2.33539465e+00, 1.54937129e+00, -2.05467527e+00, 1.89861153e+00, 3.24639277e-01, 4.90261325e-01, 2.61687350e+00, -3.93913469e-01, -6.52368818e-02, 9.96782801e-01, 7.80924006e-01, -2.21877150e+00, -8.76936173e-01, 8.10740308e-01, 2.51373517e+00, -2.45239968e+00, -2.57350725e+00, -5.17730183e-01, -1.70615298e+00, 1.18191789e-01, 3.51084133e-01, -1.57799016e-01, -1.70476727e+00, -1.56336669e+00, -1.69957048e+00, 9.34142269e-01, 2.07836070e+00, 2.58278636e+00, 1.80183310e+00, 1.08435507e+00, -1.75645825e+00, 1.44995835e+00, 1.18142751e+00, -7.09443154e-01, 2.10071475e+00, 1.68667087e-01, 2.35805441e+00, 1.92335625e+00, -1.62852828e+00, 1.92374382e+00, -2.16863145e+00, -1.05195029e-01, -2.10713666e+00, -3.10235670e-01, 9.95738834e-01, 1.73419455e+00, 9.72124549e-01, 1.26907457e+00, 2.32754197e+00, -2.14994825e+00, -2.11457926e+00, -1.04441494e+00, -2.80875702e+00, 1.12657737e+00, 1.58838027e+00, -8.99943488e-01, 6.96045664e-01, 2.75099915e+00, -1.25368430e+00, 1.12007078e+00, -2.74663062e+00, -2.62497821e+00, 1.88639217e+00, -3.80289825e-01, -6.47589408e-01, -1.01611392e+00, -2.54047030e-01, 2.01211291e+00, -1.99386743e-01, -4.72284073e-01, 4.25109318e-01, 1.43433574e+00, -2.66949955e+00, 7.45083894e-01, 7.26361966e-01, -2.99333681e+00, 6.17361038e-01, -2.75888753e+00, 5.14502201e-01, -3.09723452e-01, -8.77618394e-01, 2.60115141e+00, -1.39646441e+00, -1.31220372e+00, 2.94357819e-02, 8.80836712e-01, 2.82344057e+00, -2.78474712e+00, -4.60800803e-01, 5.29872532e-01, 1.39021370e+00, -1.94693356e+00, 6.32327077e-02, 1.70757504e-03, 2.35660579e+00, -7.03903755e-01, 9.77183021e-01, -2.71913064e+00, 1.50733970e+00, -7.87288246e-01, 2.68817182e+00, -9.16648691e-01, 1.10335919e+00, 1.94780929e+00, -8.24820346e-02, 2.90505523e+00, 6.22884229e-01, 1.55774899e+00, 1.10710017e+00, 2.59449736e+00, 2.69871283e+00, 2.94066677e+00, -2.24306770e+00, 2.85960370e+00, -1.62536606e+00, -1.88366146e+00, 5.52685321e-02, 1.82986096e-01, -1.30416455e+00, -1.13903079e+00, -1.14069374e+00, -1.40166668e+00, 5.60586905e-01, -1.69719377e+00, -5.85702494e-01, -1.26863256e+00, -1.95695511e+00};

float correct_raw[] = {-1.53458012e-01, -7.65405332e+00, 4.98830497e+00, 2.04852002e+00, -3.82358333e+00, 7.36837003e+00, 8.39924023e+00, -4.62756444e+00, -5.50727587e+00, 1.20214505e+01, -1.53043815e+01, 1.11244327e+00, 7.04805004e-01, 2.29797806e-01, -7.32889433e+00, 1.47271564e+01, 1.01044780e+01, 2.41905064e+00, -5.31388064e+00, -3.27185811e+00, -5.79821618e+00, 3.00265788e-01, 1.47316324e+00, -4.31265309e+00, -1.16285290e+01, -5.65152968e+00, -1.26282823e+00, -7.59112465e+00, -8.72511524e+00, -5.25022006e+00, -7.82986638e-01, 6.10214152e+00, 1.61352078e+01, -4.04867987e-01, -1.31349268e+01, -2.44915779e-01, 1.39773955e+01, 3.66806225e+00, 5.80719452e+00, 6.22372275e+00, 1.61941934e+00, 1.61922030e+00, 4.08148270e+00, -5.31495425e+00, 3.31723523e+00, 1.48087048e+01, 5.12534403e+00, -8.85206695e+00, 3.47283117e+00, 8.13983753e+00, -6.09539078e-02, -2.29926111e+00, 5.15988338e+00, -1.18856035e+01, -7.12207532e+00, 6.25921427e-01, 2.04809803e+00, 2.17005161e+00, 3.07331189e+00, 1.32922857e+00, -9.02002841e+00, -3.46313153e+00, -1.68674612e+00, -5.82162868e+00, 6.27101890e+00, 3.68211379e+00, 7.06585228e-01, 1.61090449e+00, 5.32800254e+00, 5.92726186e+00, 7.48984518e+00, -3.15173559e+00, -4.97384231e-01, 6.95062311e+00, 1.38288414e+01, -3.14463537e+00, -2.77009473e+00, 5.24369450e+00, 7.55776522e+00, -1.11608027e+01, 6.57839023e+00, -4.40262046e+00, 1.09980762e+00, 1.40552491e+00, -1.92546614e+01, -1.08176358e+00, 1.14700068e+00, 2.91672461e+00, -3.09855249e+00, -8.76800761e-01, -1.50497319e+01, 3.63177760e+00, 5.45905926e+00, 1.92832976e-01, -1.20174465e+01, -1.99778930e+00, -4.32951502e+00, 1.49937736e+00, 2.32108850e+00, -3.67743356e+00, -3.04202120e+00, -1.11109333e+00, -2.60361444e-01, 3.16173660e-01, 2.45348885e+00, -9.83839821e+00, -1.07808297e+01, 9.32800709e+00, -2.67422968e+00, -5.28735861e+00, -5.76634320e+00, 5.89462610e+00, -9.91125411e-01, 1.34330406e+01, -1.49107008e+01, -5.03363614e+00, 1.12514116e+00, -3.32207401e+00, 6.11599114e+00, 1.11681283e+00, 4.11900416e+00, -4.21087862e+00, 4.36223246e+00, -7.76174330e+00, 1.45358128e+00, 4.93798049e+00, 1.13569034e+01, 4.94783663e+00, -4.44187478e+00, -3.27488160e+00, 6.61695509e+00, -8.27047693e-01, 3.13530944e+00, -2.92434481e+00, -4.27918564e-01, -1.18010988e+01, 1.06067542e+00, -6.99189526e+00, 7.01859887e+00, 7.34324150e+00, 2.28063152e+00, 1.16676194e+00, -6.16082444e+00, -4.55011217e+00, -1.55240377e+01, -6.96617531e+00, 7.84952584e-03, -6.58186876e+00, -1.15664570e+00, 1.61505289e+00, 1.23032964e+00, -2.69841882e+00, 2.99208490e+00, 3.77462452e+00, -2.27055021e+00, 1.14624322e+00, -1.64543438e+00, 2.79936201e+00, -1.79692494e+01, -8.04918516e+00};

int main(int argc, char** argv) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  uint64_t cpu_device = ms.CreateCpuDevice();
  uint64_t gpu_device = ms.CreateGpuDevice(0);
  Scale input_size{8, 6, 3, 2};
  Scale weight_size{5, 3, 3, 5};
  Scale correct_size{4, 4, 5, 2};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> weight_ptr(new float[weight_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  memcpy(weight_ptr.get(), weight_raw, weight_size.Prod() * sizeof(float));
  {
    ms.current_device_id_ = cpu_device;
    ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
    Filter weight = NArray::MakeNArray(weight_size, weight_ptr);
    NArray bias = NArray::Zeros({5});
    ConvInfo conv_info{0, 0, 1, 1};
    ms.current_device_id_ = gpu_device;
    ImageBatch output = Convolution::ConvForward(input, weight, bias, conv_info);
    auto output_ptr = output.Get();
    for (int i = 0; i < correct_size.Prod(); ++i) {
      cout << i << ": " << output_ptr.get()[i] - correct_raw[i] << endl;
    }
  }
  ms.dag_scheduler().GCNodes();
  cout << ms.physical_dag().PrintDag<ExternRCPrinter>() << endl;
  ms.Finalize();
}

