#include <minerva.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;
using namespace minerva;

#define MaxV 10005
#define MaxL 200

map<string, int> wids;
vector<vector<int> > train_sents, test_sents;
vector<vector<NArray> > data_batches, target_batches, mask_batches;
vector<vector<vector<int> > > index_batches;
vector<vector<int> > actual_size;

NArray Softmax(NArray m)
{
	NArray maxes = m.Max(0);
	NArray e = Elewise::Exp(m - maxes);
	return e / e.Sum(0);
}

void Show(NArray A)
{
	int N = A.Size(0);
	int M = A.Size(1);
	auto a = A.Get();
	float* a_ptr = a.get();
	for (int i = 0; i < N; ++ i)
	{
		for (int j = 0; j < M; ++ j)
			printf("%.8f	", a_ptr[j * N + i]);
		printf("\n");
	}
}

class LSTMModel
{
	public:
		LSTMModel(int vocab_size, int input_size, int hidden_size)
		{
			Layers[0] = input_size;
			Layers[1] = hidden_size;
			Layers[2] = vocab_size;

			ig_weight_data = NArray::Randn({Layers[1], Layers[0]}, 0.0, 1);
			fg_weight_data = NArray::Randn({Layers[1], Layers[0]}, 0.0, 1);
			og_weight_data = NArray::Randn({Layers[1], Layers[0]}, 0.0, 1);
			ff_weight_data = NArray::Randn({Layers[1], Layers[0]}, 0.0, 1);

			ig_weight_prev = NArray::Randn({Layers[1], Layers[1]}, 0.0, 1);
			fg_weight_prev = NArray::Randn({Layers[1], Layers[1]}, 0.0, 1);
			og_weight_prev = NArray::Randn({Layers[1], Layers[1]}, 0.0, 1);
			ff_weight_prev = NArray::Randn({Layers[1], Layers[1]}, 0.0, 1);

			ig_weight_cell = NArray::Randn({Layers[1], 1}, 0.0, 1);
			fg_weight_cell = NArray::Randn({Layers[1], 1}, 0.0, 1);
			og_weight_cell = NArray::Randn({Layers[1], 1}, 0.0, 1);
			ff_weight_cell = NArray::Randn({Layers[1], 1}, 0.0, 1);
/*
			ig_weight_data = NArray::Ones({Layers[1], Layers[0]});
			fg_weight_data = NArray::Ones({Layers[1], Layers[0]});
			og_weight_data =NArray::Ones({Layers[1], Layers[0]});
			ff_weight_data = NArray::Ones({Layers[1], Layers[0]});

			ig_weight_prev = NArray::Ones({Layers[1], Layers[1]});
			fg_weight_prev = NArray::Ones({Layers[1], Layers[1]});
			og_weight_prev = NArray::Ones({Layers[1], Layers[1]});
			ff_weight_prev = NArray::Ones({Layers[1], Layers[1]});

			ig_weight_cell = NArray::Ones({Layers[1], 1});
			fg_weight_cell = NArray::Ones({Layers[1], 1});
			og_weight_cell = NArray::Ones({Layers[1], 1});
			ff_weight_cell = NArray::Ones({Layers[1], 1});
*/
			ig_weight_bias = NArray::Zeros({Layers[1], 1});
			fg_weight_bias = NArray::Zeros({Layers[1], 1});
			og_weight_bias = NArray::Zeros({Layers[1], 1});
			ff_weight_bias = NArray::Zeros({Layers[1], 1});

			decoder_weights = NArray::Randn({Layers[2], Layers[1]}, 0.0, 1);
			//decoder_weights = NArray::Ones({Layers[2], Layers[1]});
			decoder_bias = NArray::Zeros({Layers[2], 1});

			emb_weight = NArray::Randn({Layers[0], Layers[2]}, 0.0, 1);
			//emb_weight = NArray::Ones({Layers[0], Layers[2]});

			printf("model size: [%d %d %d].\n", Layers[0], Layers[1], Layers[2]);
		}

		void train(vector<vector<NArray> > batches, vector<vector<NArray> > targets, int words, int num_epochs, float learning_rate = 0.1)
		{
			int E = Layers[0];
			int N = Layers[1];
			int K = Layers[2];
			int batch_size = batches[0][0].Size(1);
			ms.wait_for_all();
			time_t last_time = time(NULL);
			for (int epoch_id = 0; epoch_id < num_epochs; ++ epoch_id)
			{
				float epoch_ll = 0;
				for (unsigned int b = 0; b < batches.size(); ++ b)
				{
					auto batch = batches[b];
					vector<int> this_actual_size = actual_size[b];
					int Tau = batch.size();
					int this_size = batch[0].Size(1);
					float sent_ll = 0;

					NArray data[MaxL], Hout[MaxL];
					Hout[0] = NArray::Zeros({N, this_size});
					NArray act_ig[MaxL], act_fg[MaxL], act_og[MaxL], act_ff[MaxL];
					NArray C[MaxL], dY[MaxL];
					C[0] = NArray::Zeros({N, this_size});

					NArray dBd = NArray::Zeros({K, 1});
					NArray dWd = NArray::Zeros({K, N});
					NArray dHout[MaxL], dEmb[MaxL];

					for (int t = 1; t < Tau; ++ t)
					{
						act_ig[t] = ig_weight_data * batch[t - 1] + ig_weight_prev * Hout[t - 1] + Elewise::Mult(C[t - 1], ig_weight_cell) + ig_weight_bias;
						act_ig[t] = Elewise::SigmoidForward(act_ig[t]);

						act_fg[t] = fg_weight_data * batch[t - 1] + fg_weight_prev * Hout[t - 1] + Elewise::Mult(C[t - 1], fg_weight_cell) + fg_weight_bias;
						act_fg[t] = Elewise::SigmoidForward(act_fg[t]);

						act_ff[t] = ff_weight_data * batch[t - 1] + ff_weight_prev * Hout[t - 1] + ff_weight_bias;
						act_ff[t] = Elewise::TanhForward(act_ff[t]);

						C[t] = Elewise::Mult(act_ig[t], act_ff[t]) + Elewise::Mult(C[t - 1], act_fg[t]);

						act_og[t] = og_weight_data * batch[t - 1] + og_weight_prev * Hout[t - 1] + Elewise::Mult(C[t], og_weight_cell) + og_weight_bias;
						act_og[t] = Elewise::SigmoidForward(act_og[t]);

						Hout[t] = Elewise::Mult(Elewise::TanhForward(C[t]), act_og[t]);

						NArray Y = Softmax(decoder_weights * Hout[t] + decoder_bias);

						//dY[t] = Elewise::Mult(Y - targets[b][t], mask_batches[b][t]);
						dY[t] = Y - targets[b][t];

						//cout << Y.Size() << ' ' << index_batches[b][t].size() << endl;
						//NArray Output = Y.Select(index_batches[b][t]);
						//cout << Output.Size() << endl;
						/*
						auto y_ptr = Y.Get();
						float *output_ptr = y_ptr.get();
						for (size_t s = batch_size * b; s < batch_size * b + this_size; ++ s)
						{
							float output;
							if (t < train_sents[s].size())
								output = output_ptr[K * b + train_sents[s][t]];
							else
								output = output_ptr[K * b + wids["eos"]];
							if (output < 1e-20) output = 1e-20;
							sent_ll += log2(output);
						}
						*/
					}

					NArray weight_update_ig_data = NArray::Zeros({N, E});
					NArray weight_update_ig_prev = NArray::Zeros({N, N});
					NArray weight_update_ig_cell = NArray::Zeros({N, 1});
					NArray weight_update_ig_bias = NArray::Zeros({N, 1});

					NArray weight_update_fg_data = NArray::Zeros({N, E});
					NArray weight_update_fg_prev = NArray::Zeros({N, N});
					NArray weight_update_fg_cell = NArray::Zeros({N, 1});
					NArray weight_update_fg_bias = NArray::Zeros({N, 1});

					NArray weight_update_og_data = NArray::Zeros({N, E});
					NArray weight_update_og_prev = NArray::Zeros({N, N});
					NArray weight_update_og_cell = NArray::Zeros({N, 1});
					NArray weight_update_og_bias = NArray::Zeros({N, 1});

					NArray weight_update_ff_data = NArray::Zeros({N, E});
					NArray weight_update_ff_prev = NArray::Zeros({N, N});
					NArray weight_update_ff_bias = NArray::Zeros({N, 1});

					NArray last_dHout = NArray::Zeros({N, this_size});

					for (int t = Tau - 1; t; -- t)
					{
						int sz = this_actual_size[t];
						dBd += dY[t].Sum(1) / sz;
						dWd += dY[t] * Hout[t].Trans() / sz;
						dHout[t] = decoder_weights.Trans() * dY[t] + last_dHout;
						NArray tanhC = Elewise::TanhForward(C[t]);
						NArray sen_og = Elewise::Mult(dHout[t], tanhC);
						NArray dC = Elewise::Mult((1 - Elewise::Mult(tanhC, tanhC)), Elewise::Mult(dHout[t], act_og[t]));

						// BP from og
						sen_og = Elewise::Mult(Elewise::Mult(act_og[t], (1.0 - act_og[t])), sen_og);
						dEmb[t] = og_weight_data.Trans() * sen_og / sz;
						dC += og_weight_cell.Trans() * sen_og;

						// BP from fg controled gate
						NArray sen_fg = Elewise::Mult(C[t - 1], dC);

						// BP from ig controled gate
						NArray sen_ig = Elewise::Mult(act_ff[t], dC);
						NArray sen_ff = Elewise::Mult((1 - Elewise::Mult(act_ff[t], act_ff[t])), Elewise::Mult(act_ig[t], dC));
						last_dHout = ff_weight_prev.Trans() * sen_ff;
						dEmb[t] += ff_weight_data.Trans() * sen_ff / sz;

						// BP from fg
						sen_fg = Elewise::Mult(Elewise::Mult(act_fg[t], (1.0 - act_fg[t])), sen_fg);
						dEmb[t] += fg_weight_data.Trans() * sen_fg / sz;

						// BP from ig
						sen_ig = Elewise::Mult(Elewise::Mult(act_ig[t], (1.0 - act_ig[t])), sen_ig);
						dEmb[t] += ig_weight_data.Trans() * sen_ig / sz;
						//if (b == 1 && t == 2)
						//	Show(dEmb[t]);

						// derivatives on weight matrix and bias
						weight_update_ig_data += sen_ig * batch[t - 1].Trans() / sz;
						weight_update_ig_prev += sen_ig * Hout[t - 1].Trans() / sz;
						weight_update_ig_cell += Elewise::Mult(sen_ig, C[t - 1]).Sum(1) / sz;
						weight_update_ig_bias += sen_ig.Sum(1) / sz;

						weight_update_fg_data += sen_fg * batch[t - 1].Trans() / sz;
						weight_update_fg_prev += sen_fg * Hout[t - 1].Trans() / sz;
						weight_update_fg_cell += Elewise::Mult(sen_fg, C[t - 1]).Sum(1) / sz;
						weight_update_fg_bias += sen_fg.Sum(1) / sz;

						weight_update_og_data += sen_og * batch[t - 1].Trans() / sz;
						weight_update_og_prev += sen_og * Hout[t - 1].Trans() / sz;
						weight_update_og_cell += Elewise::Mult(sen_og, C[t]).Sum(1) / sz;
						weight_update_og_bias += sen_og.Sum(1) / sz;

						weight_update_ff_data += sen_ff * batch[t - 1].Trans() / sz;
						weight_update_ff_prev += sen_ff * Hout[t - 1].Trans() / sz;
						weight_update_ff_bias += sen_ff.Sum(1) / sz;
					}

					float rate = learning_rate;
					ig_weight_prev -= rate * weight_update_ig_prev;
					ig_weight_data -= rate * weight_update_ig_data;
					ig_weight_cell -= rate * weight_update_ig_cell;
					ig_weight_bias -= rate * weight_update_ig_bias;

					fg_weight_prev -= rate * weight_update_fg_prev;
					fg_weight_data -= rate * weight_update_fg_data;
					fg_weight_cell -= rate * weight_update_fg_cell;
					fg_weight_bias -= rate * weight_update_fg_bias;

					og_weight_prev -= rate * weight_update_og_prev;
					og_weight_data -= rate * weight_update_og_data;
					og_weight_cell -= rate * weight_update_og_cell;
					og_weight_bias -= rate * weight_update_og_bias;

					ff_weight_prev -= rate * weight_update_ff_prev;
					ff_weight_data -= rate * weight_update_ff_data;
					ff_weight_bias -= rate * weight_update_ff_bias;

					decoder_weights -= rate * dWd;
					decoder_bias -= rate * dBd;

					for (int t = 1; t < Tau; ++ t)
						//	cout << dEmb[t].Size() << endl;
						//	emb_weight[sent[t - 1]] -= rate * dEmb[t];
						emb_weight.SelectiveSub(rate * dEmb[t], index_batches[b][t - 1]);

					epoch_ll += sent_ll;

					ms.wait_for_all();
				}

				float epoch_ent = epoch_ll * (-1) / words;
				float epoch_ppl = exp2(epoch_ent);
				ms.wait_for_all();
				auto current_time = time(NULL);
				printf("Epoch %d PPL = %f\n", epoch_id + 1, epoch_ppl);
				cout << "time consumed: " << difftime(current_time, last_time) << " seconds" << endl;
				current_time = last_time;
			}
		} // end of void train()

		void test(vector<vector<int> > sents, int words)
		{
			int E = Layers[0];
			int N = Layers[1];
			int K = Layers[2];
			float test_ll = 0;

			for (unsigned int s = 0; s < sents.size(); ++ s)
			{
				NArray data, Hout[MaxL];
				Hout[0] = NArray::Zeros({N, 1});
				NArray act_ig, act_fg, act_og, act_ff;
				NArray C[MaxL];
				C[0] = NArray::Zeros({N, 1});

				vector<int> sent = sents[s];
				int Tau = sent.size();
				float sent_ll = 0;
				for (int t = 1; t < Tau; ++ t)
				{
					vector<int> idx;
					idx.clear();
					idx.push_back(sent[t - 1]);
					data = emb_weight.Select(idx);
					float *target_ptr = new float[K];
					memset(target_ptr, 0, K * sizeof(float));
					target_ptr[sent[t]] = 1;
					NArray target = NArray::MakeNArray({K, 1}, shared_ptr<float> (target_ptr));

					act_ig = ig_weight_data * data + ig_weight_prev * Hout[t - 1] + Elewise::Mult(ig_weight_cell, C[t - 1]) + ig_weight_bias;
					act_ig = Elewise::SigmoidForward(act_ig);

					act_fg = fg_weight_data * data + fg_weight_prev * Hout[t - 1] + Elewise::Mult(fg_weight_cell, C[t - 1]) + fg_weight_bias;
					act_fg = Elewise::SigmoidForward(act_fg);

					act_ff = ff_weight_data * data + ff_weight_prev * Hout[t - 1] + ff_weight_bias;
					act_ff = Elewise::TanhForward(act_ff);

					C[t] = Elewise::Mult(act_ig, act_ff) + Elewise::Mult(act_fg, C[t - 1]);

					act_og = og_weight_data * data + og_weight_prev * Hout[t - 1] + Elewise::Mult(og_weight_cell, C[t]) + og_weight_bias;
					act_og = Elewise::SigmoidForward(act_og);

					Hout[t] = Elewise::Mult(act_og, Elewise::TanhForward(C[t]));

					NArray Y = Softmax(decoder_weights * Hout[t] + decoder_bias);

					auto y_ptr = Y.Get();
					float *output_ptr = y_ptr.get();
					float output = output_ptr[sent[t]];
					if (output < 1e-20) output = 1e-20;
					sent_ll += log2(output);
				}
				test_ll += sent_ll;
			}
			auto test_ent = test_ll * (-1) / words;
			auto test_ppl = exp2(test_ent);
			printf("Test PPL = %f\n", test_ppl);
		} // end of void test()

		NArray getEmbedding(vector<int> Idx)
		{
			return emb_weight.Select(Idx);
		}

		int getLayer(int idx)
		{
			return Layers[idx];
		}

	private:
		int Layers[3];
		NArray ig_weight_data, fg_weight_data, og_weight_data, ff_weight_data;
		NArray ig_weight_prev, fg_weight_prev, og_weight_prev, ff_weight_prev;
		NArray ig_weight_cell, fg_weight_cell, og_weight_cell, ff_weight_cell;
		NArray ig_weight_bias, fg_weight_bias, og_weight_bias, ff_weight_bias;
		NArray decoder_weights, decoder_bias;
		NArray emb_weight;

		MinervaSystem& ms = MinervaSystem::Instance();
};

void ReadData(int &train_words, int &test_words)
{
	train_words = test_words = 0;
	wids.clear();
	string bos, eos, unk;
	bos = "-bos-";
	eos = "-eos-";
	unk = "<unk>";
	wids[bos] = 0;
	wids[eos] = 1;
	wids[unk] = 2;

	freopen("/home/cs_user/minerva/owl/apps/train", "r", stdin);
	char s[500];
	int pointer = 2;

	for (; gets(s); )
	{
		int L = strlen(s);
		vector<int> sent;
		sent.push_back(wids[bos]);
		string w = "";
		for (int i = 0; i < L; ++ i)
			if (s[i] == ' ')
			{
				if (wids.find(w) == wids.end())
				{
					wids[w] = pointer;
					sent.push_back(pointer ++);
				}
				else
					sent.push_back(wids[w]);
				w = "";
			}
			else w += s[i];
		if (w.length() > 0)
			if (wids.find(w) == wids.end())
			{
				wids[w] = pointer;
				sent.push_back(pointer ++);
			}
			else
				sent.push_back(wids[w]);
		sent.push_back(wids[eos]);
		train_sents.push_back(sent);
		train_words += sent.size() - 1;
	}

	freopen("/home/cs_user/minerva/owl/apps/test", "r", stdin);
	for (; gets(s); )
	{
		int L = strlen(s);
		vector<int> sent;
		sent.push_back(wids[bos]);
		string w = "";
		for (int i = 0; i < L; ++ i)
			if (s[i] == ' ')
			{
				if (wids.find(w) == wids.end())
					sent.push_back(wids[unk]);
				else
					sent.push_back(wids[w]);
				w = "";
			}
			else w += s[i];
		if (w.length() > 0)
			if (wids.find(w) == wids.end())
				sent.push_back(wids[unk]);
			else
				sent.push_back(wids[w]);
		sent.push_back(wids[eos]);
		test_sents.push_back(sent);
		test_words += sent.size() - 1;
	}

	printf("read %d train samples, %d test samples.\n", train_sents.size(), test_sents.size());
	printf("vocabulary size: %d\n", wids.size());
	printf("%d train words, %d test words in total.\n", train_words, test_words);
}

NArray Eye(int N)
{
	float* eye_ptr = new float[N * N];
	memset(eye_ptr, 0, sizeof(eye_ptr));
	for (int i = 0; i < N; ++ i)
		eye_ptr[i * N + i] = 1;
	return NArray::MakeNArray({N, N}, shared_ptr<float> (eye_ptr));
}

void MakeBatches(LSTMModel model, int batch_size)
{
	MinervaSystem& ms = MinervaSystem::Instance();

	data_batches.clear();
	target_batches.clear();
	index_batches.clear();
	actual_size.clear();
	mask_batches.clear();

	int num_batches = (train_sents.size() - 1) / batch_size + 1;
	NArray I = Eye(model.getLayer(2));
	for (int b = 0; b < num_batches; ++ b)
	{
		size_t this_size = batch_size;
		if (b == num_batches - 1) this_size = train_sents.size() - batch_size * b;
		size_t max_len = 0;
		for (size_t s = b * batch_size; s < b * batch_size + this_size; ++ s)
			max_len = max(max_len, train_sents[s].size());

		vector<NArray> data_batch, target_batch, mask_batch;
		vector<vector<int> > index_batch;
		vector<int> this_actual_size;

		data_batch.clear();
		target_batch.clear();
		index_batch.clear();
		this_actual_size.clear();
		mask_batch.clear();
		for (size_t k = 0; k < max_len; ++ k)
		{
			int o_size = model.getLayer(2);
			vector<int> Idx;
			int size = this_size;
			Idx.clear();
			float *mask_ptr = new float[o_size * this_size];
			for (int i = 0; i < o_size * this_size; ++ i)
				mask_ptr[i] = 1;
			ms.wait_for_all();

			for (size_t s = 0; s < this_size; ++ s)
				if (k < train_sents[batch_size * b + s].size())
					Idx.push_back(train_sents[batch_size * b + s][k]);
				else
				{
					Idx.push_back(wids["-eos-"]);
					memset(mask_ptr + s * o_size, 0, sizeof(float) * o_size);
					-- size;
				}
			//Show(model.getEmbedding(Idx));
			data_batch.push_back(model.getEmbedding(Idx));
			target_batch.push_back(I.Select(Idx));
			index_batch.push_back(Idx);
			this_actual_size.push_back(size);
			//mask_batch.push_back(NArray::MakeNArray({o_size, this_size}, shared_ptr<float> (mask_ptr)));
		}
		target_batches.push_back(target_batch);
		data_batches.push_back(data_batch);
		index_batches.push_back(index_batch);
		actual_size.push_back(this_actual_size);
		mask_batches.push_back(mask_batch);
	}
}

int H = 10;

int main(int argc, char** argv)
{
	if (argc == 2) H = atoi(argv[1]);
	MinervaSystem::Initialize(&argc, &argv);
	MinervaSystem& ms = MinervaSystem::Instance();
	ms.SetDevice(ms.device_manager().CreateGpuDevice(0));
	int train_words, test_words;
	ReadData(train_words, test_words);
	LSTMModel model(wids.size(), H, H);
	MakeBatches(model, 100);
	for (int i = 0; i < 10; ++ i)
	{
		model.train(data_batches, target_batches, train_words, 1);
		model.test(test_sents, test_words);
	}
}
