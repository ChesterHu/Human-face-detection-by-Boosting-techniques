#include<iostream>
#include<fstream>
#include<algorithm>
#include"load.h"
#include<math.h>
using namespace std;
void boost(int t);
void output();
void real_boost(int t);
int load_test(int s, int idx);
void test();

int main() {
	//Load data
	
	//cout << load(32, 0) << endl;
	//cout << dt[0][0][0] << endl;
	// Generate clf of adaboost
	
	/*for (int i = 0; i < NUM_CLF; ++i) {
		cout << clf[i][0];
	}*/
	
	load_data();
	load_clf();
	
	cout << "Start training" << endl;
	for (int t = 0; t < T; t++) {
		cout << "Iteration " << t+1 << endl;
		boost(t);
	}
	cout << "Training finish" << endl;
	
	test();
	//load_test(64, 0);
	//cout << dt[0][15][15] << endl;
	//cout << cord[0][0] << endl;
	//cout << cord[0][1] << endl;
	//cout<<load(112, 0)<<endl;
	//cout << dt[6799][0][0] << endl;
	//cout << "Training err: " << boost_predict(T) << endl;
	//output();

	
	//cout << "hehe" << endl;
	
	//cout << "Training finish" << endl;
	/*for (int i = 0; i < T; ++i) {
		cout << real_ada_clf[i] << endl;
	}*/
	//cout << "Training err: " << boost_predict(T) << endl;
	return 0;
}


void boost(int t) {
	// get the min error clf.
	int min_f = -1;
	min_f=load_score();
	/*
	else {
		double min_err = 0.5;
		int min_idx = -1;
		for (int i = 0; i < NUM_CLF; ++i) {
			if (clf[i][0] != CLF::empty&&clf[i][6] < min_err) {
				min_idx = i;
				min_err = clf[i][6];
			}
		}
		min_f = min_idx;
	}
	*/
	for (int i = 0; i < 7; ++i) ada_clf[t][i] = clf[min_f][i];
	cout << "clf minimum error: " << clf[min_f][6] << endl;
	clf[min_f][0] = CLF::empty;
	// Assign weight for the new clf
	alpha[t] = 0.5*log((1 - clf[min_f][6]) / clf[min_f][6]);
	// Update the weights of the data points
	double total = 0;
	double new_W[TRAIN_FACE + TRAIN_NONFACE];
	for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
		if ((feature_score[min_f][i].value < clf[min_f][5]&&feature_score[min_f][i].index<TRAIN_FACE)
			||((feature_score[min_f][i].value > clf[min_f][5] && feature_score[min_f][i].index>TRAIN_FACE)))
			new_W[feature_score[min_f][i].index] = W[feature_score[min_f][i].index]*exp(-alpha[t]);
		else new_W[feature_score[min_f][i].index] = W[feature_score[min_f][i].index] * exp(alpha[t]);
		total +=new_W[feature_score[min_f][i].index];
	}
	// New feature score
	/*double total_err = 0;
	double err = 1;
	for (int i = 0; i < TRAIN_FACE; ++i) {
		total_err += new_W[i]/total;
	}
	for (int i = 0; i < NUM_CLF; ++i) {
		if (clf[i][0] == CLF::empty) continue;
		for (int j = 0; j < TRAIN_FACE + TRAIN_NONFACE; ++j) {
			feature_score[i][j].value *= new_W[feature_score[i][j].index] / W[feature_score[i][j].index] / total;
		}
		sort(feature_score[i], feature_score[i] + TRAIN_FACE + TRAIN_NONFACE, cmp);
		double err_count = total_err;
		for (int k = 0; k < TRAIN_FACE + TRAIN_NONFACE; ++k)
		{
			if (feature_score[i][k].index < TRAIN_FACE) err_count -= W[feature_score[i][k].index];
			else err_count += W[feature_score[i][k].index];
			if (err_count < clf[i][6])
			{
				clf[i][6] = err_count;
				clf[i][5] = feature_score[i][i].value;
			}
		}
		if (clf[i][6] < err)
		{
			err = clf[i][6];
			min_f = i;
		}
	}*/
	for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
		W[i] =new_W[i]/total;
	}
}

void real_boost(int t) {
	int min_f = -1;
	// Get the min Z
	if(t==0) min_f = real_load_score();
	else {
		double min_err = 10e7;
		int min_idx = -1;
		for (int i = 0; i < NUM_CLF; ++i) {
			if (clf[i][0] != CLF::empty&&clf[i][6] < min_err) {
				min_idx = i;
				min_err = clf[i][6];
			}
		}
		min_f = min_idx;
	}
	// Store the index.
	real_ada_clf[t] = min_f;
	//cout << clf[min_f][6] << endl;
	for (int i = 0; i < 7;++i) ada_clf[t][i] = clf[min_f][i];
	clf[min_f][0] = CLF::empty;

	// Reweight
	// Compute h
	double bin_size = (feature_score[min_f][TRAIN_FACE + TRAIN_NONFACE-1].value - feature_score[min_f][0].value) / B;
	//cout << feature_score[min_f][TRAIN_FACE + TRAIN_NONFACE-1].value << endl;
	//cout << feature_score[min_f][0].value << endl;
	//cout << bin_size << endl;
	int l = 0;
	for (int b = 0; b < B; ++b) {
		double p = 0;
		double q = 0;
		for (int i = l; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
			if (feature_score[min_f][i].value > (b + 1)*bin_size) break;
			if (feature_score[min_f][i].index < TRAIN_FACE) p += W[feature_score[min_f][i].index];
			else q += W[feature_score[min_f][i].index];
			++l;
		}
		if (p < 10e-9 && q < 10e-9) {
			p = 1;
			q = 1;
		}
		else if (p < 10e-9 && q != 0) {
			h[min_f][b] = -1*q;
		}
		else if (p != 0 && q <10e-9) {
			h[min_f][b] = 1*p;
		}
		else h[min_f][b] = 0.5*log( p / q );
		//cout << h[min_f][b] << endl;
	}


	// Update weight
	double w_sum = 0;
	int k = 0;
	for (int b = 0; b < B; ++b) {
		for (int i = k; i < TRAIN_FACE+TRAIN_NONFACE; ++i) {
			//real_f_score[t][i] = feature_score[min_f][i];
			if (feature_score[min_f][i].value >(b + 1)*bin_size) break;
			if (feature_score[min_f][i].index < TRAIN_FACE) W[feature_score[min_f][i].index] *= exp(-h[min_f][b]);
			else W[feature_score[min_f][i].index] *= exp(h[min_f][b]);
			//cout << W[feature_score[min_f][i].index] << endl;
			w_sum += W[feature_score[min_f][i].index];
			++k;
		}
	}
	// Normalize the weight
	for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
		W[i] =W[i]/w_sum;
		//cout << W[i] << endl;
	}
	/*for (int i = 0; i < B; ++i) {
		cout << h[min_f][i] << endl;
	}*/
}


void output() {
	/*ofstream os_model("C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\result\\clf_adaboost.txt", ios::app);
	if (os_model)
	{
		for (int i = 0; i<T; ++i)
		{
			if (clf[i][0] != 0) {
				for (int j = 0; j < 7; ++j)
					os_model << clf[i][j] << " ";
				os_model << endl;
			}
		}
	}
	os_model.close();
	ofstream os_weight("C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\result\\weight.txt", ios::app);
	if (os_weight)
	{
		for (int i = 0; i<T; ++i)
		{
			os_weight << alpha[i] << endl;
		}
	}
	os_weight.close();
	*/
	ofstream os_pre("C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\result\\pre50.txt", ios::app);
	if (os_pre)
	{
		for (int i = 0; i<TRAIN_FACE+TRAIN_NONFACE; ++i)
		{
			
			os_pre << predict(T,dt[i]) << endl;
		}
	}
	os_pre.close();
}

void load_model() {
	ifstream in("C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\result\\pre.txt");
}

void test() {
	// Load all data
	int n_dt = 0;// count for test data
	//n_dt = load_test(64, n_dt);
	n_dt = load_test(164, n_dt);
	//n_dt = load_test(112, n_dt);
	/*
	cout << dt[n_dt - 1][15][15] << endl;
	cout << n_dt<< endl;
	
	cout << dt[n_dt - 1][15][15] << endl;
	cout << n_dt << endl;
	*/
	//n_dt = load(128, n_dt);
	//cout << n_dt << endl;
	//n_dt is the num of files. The last file is n_dt-1
	//cout << test_dt[n_dt-1][0][0] << endl;
	//cout << cord[n_dt-1][0] << " " << cord[n_dt-1][1] << " " << cord[n_dt-1][2] << endl;
	// prediction of boost.
	ofstream os_pre("C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\Test_and_background_Images\\face164.txt");
	if (os_pre)
	{
		//cout << "hehe" << endl;
		for (int i = 0; i<n_dt; ++i)
		{
			//cout << "hehe" << endl;
			os_pre << predict(T, dt[i]) << ' ' << cord[i][0] << ' ' << cord[i][1] << ' ' << cord[i][2] << endl;
		}
	}
	os_pre.close();
}


// Load file and return the last file idx(+1)
int load_test(int s,int idx) {
	// s: width & height
	// idx: start idx to store data
	int w = s; int h = s;
	int c = 5312; int r = 2988;
	// Start point.
	int x = 0;
	if (s == 96) {
		x = 1000;
		r -= 688;
	}
	if (s == 102) {
		x = 1000;
		r -= 1000;
	}
	if (s == 164) {
		x = 1500;
		r -= 688;
	}
	// count
	int count = idx;
	while (x < r && (x + w) < r) {
		int y = w;
		while (y < c && (y + h) < c) {
			char path[110]; path[0] = '\0';
			sprintf_s(path, "C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\Test_and_background_Images\\split\\%d__%d_%d.txt", s, x, y);
			//cout<<path<<endl;
			ifstream in_test(path);//return double 2-dim array
							  // Load data...
			//cout << "hehe" << endl;
			if (in_test) {
				for (int i = 0; i < ROW; ++i) {
					for (int j = 0; j < COL; ++j) {
						in_test >> dt[count][i][j];
						//cout << dt[count][i][j] << " ";
					}
				}
			}
			in_test.close();
			cord[count][0] = s; cord[count][1] = x; cord[count][2] = y;
			//cout << x<<" "<<y << endl;
			count++;
			if (count >= (TRAIN_FACE+TRAIN_NONFACE)) break;
			y += 16;
		}
		x += 16;
	}
	return count;
}