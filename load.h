#include<fstream>
#include<iostream>
#include <algorithm>
#include"global.h"
#include <math.h>
using namespace std;


// int data[TRAIN_FACE + TRAIN_NONFACE][16][16];
void load_data() {
	cout << "Constructing training set..." << endl;
	for (int i = 1; i <= (TRAIN_FACE + TRAIN_NONFACE); ++i) {
		char path[110]; path[0] = '\0';
		if (i < TRAIN_FACE + 1) {
			sprintf_s(path, "C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\data_txt\\train_face\\face%d.txt", i);
			W[i - 1] = 1.0 / 2.0 / TRAIN_FACE;
		}
		else {
			sprintf_s(path, "C:\\Users\\huchu\\Documents\\Fall 2016\\M231\\P2\\data_txt\\train_nonface\\nonface%d.txt", i);
			W[i - 1] = 1.0 / 2.0 / TRAIN_NONFACE;
		}
		ifstream in(path);//return double 2-dim array
						  // Load data...
		for (int r = 0; r < ROW; ++r) {
			for (int l = 0; l < COL; ++l) {
				in >> dt[i - 1][r][l];
			}
		}
		in.close();//close 
	}
	cout << "Training set: " << endl;
	cout << "Face data: " << TRAIN_FACE << endl;
	cout << "Non-face data: " << TRAIN_NONFACE << endl;
}

void load_clf() {
	int c = 0;
	for (int x = 0; x < ROW; ++x) {
		for (int y = 0; y < COL; ++y) {
			for (int w = 4; w < ROW; ++w) {
				for (int h = 4; h < COL; ++h)
				{
					if (c >= NUM_CLF) break;
					if ((x - w) >= 0 && (y - h) >= 0)
					{
						clf[c][1] = w; clf[c][2] = h; clf[c][3] = x; clf[c][4] = y; clf[c][5] = 0; clf[c][6] = TRAIN_FACE + TRAIN_NONFACE;
						if (w % 3 == 0)
						{
							clf[c][0] = CLF::H2;
							if (c + 2 < NUM_CLF) {
								c++; clf[c][0] = CLF::V1;
								clf[c][1] = w; clf[c][2] = h; clf[c][3] = x; clf[c][4] = y; clf[c][5] = 0; clf[c][6] = TRAIN_FACE + TRAIN_NONFACE;
								c++; clf[c][0] = CLF::H1;
								clf[c][1] = w; clf[c][2] = h; clf[c][3] = x; clf[c][4] = y; clf[c][5] = 0; clf[c][6] = TRAIN_FACE + TRAIN_NONFACE;
							}
						}
						else if (h % 3 == 0)
						{
							clf[c][0] = CLF::V2;
							if (c + 2 < NUM_CLF) {
								c++; clf[c][0] = CLF::V1;
								clf[c][1] = w; clf[c][2] = h; clf[c][3] = x; clf[c][4] = y; clf[c][5] = 0; clf[c][6] = TRAIN_FACE + TRAIN_NONFACE;
								c++; clf[c][0] = CLF::H1;
								clf[c][1] = w; clf[c][2] = h; clf[c][3] = x; clf[c][4] = y; clf[c][5] = 0; clf[c][6] = TRAIN_FACE + TRAIN_NONFACE;
							}
						}
						else
						{
							clf[c][0] = CLF::V1;
							if (c + 1 < NUM_CLF) {
								c++; clf[c][0] = CLF::H1;
								clf[c][1] = w; clf[c][2] = h; clf[c][3] = x; clf[c][4] = y; clf[c][5] = 0; clf[c][6] = TRAIN_FACE + TRAIN_NONFACE;
							}
						}
						c++;
					}
				}
			}
		}
	}
}


// Return the idx of min err classifier
int load_score()
{
	int idx = 0;
	double err = 1;
	int min_idx = -1;
	// Total error
	double total_err = 0;
	for (int i = 0; i < TRAIN_FACE; ++i) {
		total_err += W[i];
	}

	while (idx < NUM_CLF)
	{
		int A_x = 0, B_x = 0, C_x = 0, D_x = 0, A_y = 0, B_y = 0, C_y = 0, D_y = 0, a_x = 0, b_x = 0, c_x = 0, d_x = 0, a_y = 0, b_y = 0, c_y = 0, d_y = 0;
		//cout << idx << endl;
		// Total block
		 A_x = clf[idx][3];					 A_y = clf[idx][4];
		 B_x = clf[idx][3] - clf[idx][1];	 B_y = A_y;
		 C_x = A_x;							 C_y = clf[idx][4] - clf[idx][2];
		 D_x = B_x;							 D_y = C_y;
		// For different clf
		switch ((int)clf[idx][0])
		{

		case CLF::empty:
		{
			break;
		}

		case CLF::V1:
		{
			// Sub block 
			 a_x = A_x;					 a_y = A_y;
			 b_x = B_x;					 b_y = B_y;
			 c_x = a_x;					 c_y = clf[idx][4] - clf[idx][2] / 2;
			 d_x = b_x;					 d_y = c_y;
			break;
		}

		case CLF::H1:
		{
			// Sub block
			 a_x = A_x;					 a_y = A_y;
			 b_x = a_x - clf[idx][1] / 2;	 b_y = a_y;
			 c_x = a_x;					 c_y = a_y - clf[idx][2];
			 d_x = b_x;					 d_y = c_y;
			break;
		}

		case CLF::V2:
		{
			// Sub block
			 a_x = A_x;						 a_y = A_y - clf[idx][2] / 3;
			 b_x = B_x;						 b_y = a_y;
			 c_x = a_x;						 c_y = C_x + clf[idx][2] / 3;
			 d_x = b_x;						 d_y = c_y;
			break;
		}

		case CLF::H2:
		{
			// Sub block 
			 a_x = A_x - clf[idx][1] / 3;	 a_y = A_y;
			 b_x = B_x + clf[idx][1] / 3;	 b_y = a_y;
			 c_x = a_x;						 c_y = C_y;
			 d_x = b_x;						 d_y = c_y;
			break;
		}
		default:
		{
			break;
		}
		}
		while (clf[idx][0] == CLF::empty) {
			idx++;
			continue;
		}
			
		if (idx >= NUM_CLF) break;
		// Get score
		for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i)
		{
			feature_score[idx][i].value = (dt[i][A_y][A_x] - dt[i][B_y][B_x] - dt[i][C_y][C_x] + dt[i][D_y][D_x])
				- 2 * (dt[i][a_y][a_x] - dt[i][b_y][b_x] - dt[i][c_y][c_x] + dt[i][d_y][d_x]);
			feature_score[idx][i].index = i;
		}
		// Find threshold
		sort(feature_score[idx], feature_score[idx] + TRAIN_FACE + TRAIN_NONFACE, cmp);
		double err_count = total_err;
		for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i)
		{
			if (feature_score[idx][i].index < TRAIN_FACE) err_count -= W[feature_score[idx][i].index];
			else err_count += W[feature_score[idx][i].index];
			if (err_count < clf[idx][6])
			{
				clf[idx][6] = err_count;
				clf[idx][5] = feature_score[idx][i].value;
			}
		}
		if (clf[idx][6] < err)
		{
			err = clf[idx][6];
			min_idx = idx;
		}
		++idx;
	}
	return min_idx;
	
}

// Given index of a clf, reweight the data and return the next min_feature.
int re_weight(int t, int idx) {
	double sum = 0;
	double new_W[TRAIN_FACE + TRAIN_NONFACE];
	for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
		if ((feature_score[idx][i].value < clf[idx][5] && i < TRAIN_FACE) || (feature_score[idx][i].value > clf[idx][5] && i > TRAIN_FACE)) new_W[i] = W[i] * exp(-alpha[t]);
		else new_W[i] = W[i] * exp(alpha[t]);
		sum += new_W[i];
	}
	for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
		new_W[i] /= sum;
	}
	double total_err = 0;
	for (int i = 0; i < TRAIN_FACE; ++i) {
		total_err += new_W[i];
	}
	// Get new score for each data set, and updata threshold
	int i = 0;
	double min_err = 0.5;
	int min_feature = -1;
	while (i < NUM_CLF)
	{
		while (clf[i][0] == CLF::empty&&i<NUM_CLF) i++;
		if (i >= NUM_CLF) break;
		for (int j = 0; j < TRAIN_FACE + TRAIN_NONFACE; ++j) {
			feature_score[i][j].value *= new_W[j] / W[j];
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
				clf[i][5] = feature_score[i][k].value;
			}
		}
		if (clf[i][6] < min_err) {
			min_err = clf[i][6];
			min_feature = i;
		}
		i++;
	}

	// get new weight
	for (int j = 0; j < TRAIN_FACE + TRAIN_FACE; ++j) {
		W[j] = new_W[j];
	}
	return min_feature;
}


// Get score
double predict(int t,int data[16][16]) {
	double threshold = 0;
	for (int i=0; i < t; ++i) {
		threshold += alpha[i];
	}
	double score = 0;
	int bin_size = (TRAIN_FACE + TRAIN_NONFACE) / B;
	for (int i = 0; i < t; ++i) {
		int A_x = 0, B_x = 0, C_x = 0, D_x = 0, A_y = 0, B_y = 0, C_y = 0, D_y = 0, a_x = 0, b_x = 0, c_x = 0, d_x = 0, a_y = 0, b_y = 0, c_y = 0, d_y = 0;
		// Total block
		 A_x = ada_clf[i][3];					 A_y = ada_clf[i][4];
		 B_x = ada_clf[i][3] - ada_clf[i][1];		 B_y = A_y;
		 C_x = A_x;								 C_y = ada_clf[i][4] - ada_clf[i][2];
		 D_x = B_x;								 D_y = C_y;
		switch ((int)ada_clf[i][0])
		{
		case CLF::V1:
		{
			// Sub block 
			 a_x = A_x;					 a_y = A_y;
			 b_x = B_x;					 b_y = B_y;
			 c_x = a_x;					 c_y = ada_clf[i][4] - ada_clf[i][2] / 2;
			 d_x = b_x;					 d_y = c_y;
			break;
		}
		case CLF::H1:
		{
			// Sub block
			 a_x = A_x;						 a_y = A_y;
			 b_x = a_x - ada_clf[i][1] / 2;	 b_y = a_y;
			 c_x = a_x;						 c_y = a_y - ada_clf[i][2];
			 d_x = b_x;						 d_y = c_y;
			break;
		}
		case CLF::V2:
		{
			// Sub block
			 a_x = A_x;						 a_y = A_y - ada_clf[i][2] / 3;
			 b_x = B_x;						 b_y = a_y;
			 c_x = a_x;						 c_y = C_x + ada_clf[i][2] / 3;
			 d_x = b_x;						 d_y = c_y;
			break;
		}
		case CLF::H2:
		{
			// Sub block 
			 a_x = A_x - ada_clf[i][1] / 3;	 a_y = A_y;
			 b_x = B_x + ada_clf[i][1] / 3;	 b_y = a_y;
			 c_x = a_x;						 c_y = C_y;
			 d_x = b_x;						 d_y = c_y;
			break;
		}
		default:
			break;
		}
		if (((data[A_y][A_x] - data[B_y][B_x] - data[C_y][C_x] + data[D_y][D_x])
			- 2 * (data[a_y][a_x] - data[b_y][b_x] - data[c_y][c_x] + data[d_y][d_x])) < ada_clf[i][5]) score += alpha[i];
		else score -= alpha[i];
	}
	return score;
}

// Get prediction error for t clfs
/*double boost_predict(int t) {
	double err = 0;
	for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i) {
		bool face = predict(t, dt[i]);
		if ((i > TRAIN_FACE&&face==true) || (i < TRAIN_FACE && face==false)) ++err;
	}
	return err / (TRAIN_FACE + TRAIN_NONFACE);
}*/



// Load score realboost
int real_load_score()
{
	int idx = 0;
	int min_idx = -1;
	// Total error
	double score = 10e7;
	
	while (idx < NUM_CLF)
	{
		int A_x = 0, B_x = 0, C_x = 0, D_x = 0, A_y = 0, B_y = 0, C_y = 0, D_y = 0, a_x = 0, b_x = 0, c_x = 0, d_x = 0, a_y = 0, b_y = 0, c_y = 0, d_y = 0;
		//cout << idx << endl;
		// Total block
		 A_x = clf[idx][3];					 A_y = clf[idx][4];
		 B_x = clf[idx][3] - clf[idx][1];	 B_y = A_y;
		 C_x = A_x;							 C_y = clf[idx][4] - clf[idx][2];
		 D_x = B_x;							 D_y = C_y;
		// For different clf
		switch ((int)clf[idx][0])
		{
		case CLF::empty:
		{
			break;
		}

		case CLF::V1:
		{
			// Sub block 
			 a_x = A_x;					 a_y = A_y;
			 b_x = B_x;					 b_y = B_y;
			 c_x = a_x;					 c_y = clf[idx][4] - clf[idx][2] / 2;
			 d_x = b_x;					 d_y = c_y;
			break;
		}

		case CLF::H1:
		{
			// Sub block
			 a_x = A_x;					 a_y = A_y;
			 b_x = a_x - clf[idx][1] / 2;	 b_y = a_y;
			 c_x = a_x;					 c_y = a_y - clf[idx][2];
			 d_x = b_x;					 d_y = c_y;
			break;
		}

		case CLF::V2:
		{
			// Sub block
			 a_x = A_x;						 a_y = A_y - clf[idx][2] / 3;
			 b_x = B_x;						 b_y = a_y;
			 c_x = a_x;						 c_y = C_x + clf[idx][2] / 3;
			 d_x = b_x;						 d_y = c_y;
			break;
		}

		case CLF::H2:
		{
			// Sub block 
			 a_x = A_x - clf[idx][1] / 3;	 a_y = A_y;
			 b_x = B_x + clf[idx][1] / 3;	 b_y = a_y;
			 c_x = a_x;						 c_y = C_y;
			 d_x = b_x;						 d_y = c_y;
			break;
		}
		default:
		{
			break;
		}
		}
		// If empty skip clf
		if (clf[idx][0] == CLF::empty) {
			idx++;
			continue;
		}
		// Get feature
		for (int i = 0; i < TRAIN_FACE + TRAIN_NONFACE; ++i)
		{
			feature_score[idx][i].value = (dt[i][A_y][A_x] - dt[i][B_y][B_x] - dt[i][C_y][C_x] + dt[i][D_y][D_x])
				- 2 * (dt[i][a_y][a_x] - dt[i][b_y][b_x] - dt[i][c_y][c_x] + dt[i][d_y][d_x]);
			feature_score[idx][i].index = i;
		}

		// Get score
		// Sort score
		sort(feature_score[idx], (feature_score[idx] + TRAIN_FACE + TRAIN_NONFACE), cmp);
		
		// Split space into B block
		double temp_score = 0;
		double bin_size = (feature_score[idx][TRAIN_FACE + TRAIN_NONFACE - 1].value - feature_score[idx][0].value) / B;
		int k = 0;
		for (int b = 0; b < B; ++b) {
			double p = 0;
			double q = 0;
			for (int i = k; i < TRAIN_FACE+TRAIN_NONFACE; ++i) {
				if (feature_score[idx][i].value > (b + 1)*bin_size) {
					break;
				}
				if (feature_score[idx][i].index < TRAIN_FACE) p += W[feature_score[idx][i].index];
				else q += W[feature_score[idx][i].index];
				++k;
			}
			temp_score +=2* sqrt(p*q);
			//h[idx][b] = 0.5*log(p / q);
		}
		clf[idx][6] = temp_score;
		if (temp_score < score) {
			score = temp_score;
			min_idx = idx;
		}
		++idx;
	}
	return min_idx;
}




double real_predict(int t, int data[16][16]) {
	double score = 0;
	for (int i = 0; i < t; ++i) {
		// Get the idx of that clf from all clf.
		int idx = real_ada_clf[i];
		int A_x = 0, B_x = 0, C_x = 0, D_x = 0, A_y = 0, B_y = 0, C_y = 0, D_y = 0, a_x = 0, b_x = 0, c_x = 0, d_x = 0, a_y = 0, b_y = 0, c_y = 0, d_y = 0;
		// Total block
		A_x = clf[idx][3];					 A_y = clf[idx][4];
		B_x = clf[idx][3] - clf[idx][1];	 B_y = A_y;
		C_x = A_x;							 C_y = clf[idx][4] - clf[idx][2];
		D_x = B_x;							 D_y = C_y;
		switch ((int)ada_clf[i][0])
		{
		case CLF::V1:
		{
			// Sub block 
			a_x = A_x;					 a_y = A_y;
			b_x = B_x;					 b_y = B_y;
			c_x = a_x;					 c_y = clf[idx][4] - clf[idx][2] / 2;
			d_x = b_x;					 d_y = c_y;
			break;
		}
		case CLF::H1:
		{
			// Sub block
			a_x = A_x;					 a_y = A_y;
			b_x = a_x - clf[idx][1] / 2;	 b_y = a_y;
			c_x = a_x;					 c_y = a_y - clf[idx][2];
			d_x = b_x;					 d_y = c_y;
			break;
		}
		case CLF::V2:
		{
			// Sub block
			a_x = A_x;						 a_y = A_y - clf[idx][2] / 3;
			b_x = B_x;						 b_y = a_y;
			c_x = a_x;						 c_y = C_x + clf[idx][2] / 3;
			d_x = b_x;						 d_y = c_y;
			break;
		}
		case CLF::H2:
		{
			a_x = A_x - clf[idx][1] / 3;	 a_y = A_y;
			b_x = B_x + clf[idx][1] / 3;	 b_y = a_y;
			c_x = a_x;						 c_y = C_y;
			d_x = b_x;						 d_y = c_y;
			break;
		}
		default:
			break;
		}
		double position = ((data[A_y][A_x] - data[B_y][B_x] - data[C_y][C_x] + data[D_y][D_x]) 
			- 2 * (data[a_y][a_x] - data[b_y][b_x] - data[c_y][c_x] + data[d_y][d_x]));
		double bin_size = (feature_score[idx][TRAIN_FACE + TRAIN_NONFACE-1].value - feature_score[idx][0].value) / B;
		// cout << "position is "<<position << endl;
		// cout << "max" << feature_score[idx][TRAIN_FACE + TRAIN_NONFACE-1].value << endl;
		// cout << "min" << feature_score[idx][0].value << endl;
		// Find the data's position in this clf.
		// cout << ada_clf[i][2] << " " << clf[idx][2] << endl;
		for (int b = 0; b < B; b++) {
			if ((feature_score[idx][0].value + bin_size*b) <= position && position <= (feature_score[idx][0].value + (b + 1)*bin_size)) {
				score += h[idx][b];
				//cout << h[idx][b]<<endl;
				//cout << b << endl;
				break;
			}
			// If exceeds the boundary
			//else if ((feature_score[idx][0].value > position)||(feature_score[idx][TRAIN_FACE+TRAIN_NONFACE-1].value<position)) cout << "?" << endl;
		}
	}
	cout << "Score is " << score << endl;
	return score;
}

