// Default parameters
const int ROW = 16;
const int COL = 16;
const int TRAIN_FACE = 11838;
const int TRAIN_NONFACE = 40000;
const int NUM_CLF = 2000;
const int T = 100;
// Num of blocks
const int B = 20;


// Data set
int dt[TRAIN_FACE + TRAIN_NONFACE][16][16];
//int test_dt[25000][16][16];
// test data's : resolution, x, y
int cord[25000][3];
// Classifiers
//		0		1    2                           3 4     5         6
// clf:{type,height,width,right_botton_cordinate(x,y),threshold,testerror} 
enum CLF { empty = 0, V1 = 1, H1 = 2, V2 = 3, H2 = 4 };
double clf[NUM_CLF][7];
//			0		1    2                            3 4     5         6	    
// real_clf:{type,height,width,right_botton_cordinate(x,y),threshold,testerror} 

// Store the index from the all clf.
int real_ada_clf[T];

double ada_clf[T][7];

// Weight for data
double W[TRAIN_FACE + TRAIN_NONFACE];
// Weight for clf
double alpha[T];

// Score of each clf
struct score
{
	double value;
	int index;
};

bool cmp(struct score a, struct score b)
{
	if (a.value < b.value)
	{
		return true;
	}
	return false;
}

double h[NUM_CLF][B];
//score real_f_score[T][TRAIN_FACE + TRAIN_NONFACE];
score feature_score[NUM_CLF][TRAIN_FACE+TRAIN_NONFACE];

