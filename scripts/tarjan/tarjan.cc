/*  Date: 2020/11/26
 *  Author: Yuchen 
 *  Descripion: tarjan.cc is used to get all the SCCs(Strongly Connected Components) of a given Graph G.
 *              Presently, Graph is represented by Adjacency Matrix. It will be changed to Adjacency List
 *              in next version.
 *              Use class to do this!!
 */


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stack>
#include <algorithm>

using namespace std;

//QAQ these will be deprecated in next version 
const int kMaxV = 7115;
const int kNumNodes = 7115;
const int kNumEdges = 103690;

// data path: maybe need to change it in Windows QAQ
string basepath = "./";

// used in function TarjanSCC
int dfn_index = 0; // (timestamp)
int graph[kMaxV][kMaxV]; // better to use Adjacency List
int low[kMaxV]; // The root number of the smallest subtree which contains Node u.
int dfn[kMaxV]; // The order in which nodes are searched during depth-first search traversal.
bool in_stack[kMaxV];
vector< vector<int> > scc_list; // save all the SCCs
stack<int> st;

// functions' declaration
void InitArrays();
void BuildGraphFromTXT(string filepath);
void Tarjan(int u);
void GetSCCs();
void SaveSCCsToTXT(string filename);

int main() {
    InitArrays();
    BuildGraphFromTXT(basepath + "wiki-vote7115.txt"); // use python to preprocess "wiki-vote7115.txt"
    GetSCCs();
    SaveSCCsToTXT(basepath + "wiki-SCCs.txt"); // "wiki-SCCs.txt"
    return 0;
}


void InitArrays() {
    for(int i = 0; i < kMaxV; i ++) {
        dfn[i] = 0;
        in_stack[i] = false;
        for(int j = 0; j < kMaxV; j ++) {
            graph[i][j] = 0;
        }
    }
}


void BuildGraphFromTXT(string filename) {
    ifstream in;
    in.open(filename, ios::in);
    if(in.is_open()){ //be cautious here: filename inexistance will cause Endless loop
        int u, v;
        int cnt = 0;
        int max_node_num = 0;
        while(!in.eof()) {
            in >> u >> v;
            graph[u][v] = 1;
            cnt++;
            max_node_num = max(u, v) > max_node_num ? max(u, v) : max_node_num;
        }
        //print reading results
        cout << "Finish Reading " + filename << endl;
        cout << "Total Nodes: " << max_node_num + 1 << endl;
        cout << "Total Edges: " << cnt << endl;
        in.close();
    } else {
        cerr << "Failed to open file " + filename << endl;
    }
    
}


void Tarjan(int u) {
    low[u] = dfn[u] = ++ dfn_index;
    st.push(u);
    in_stack[u] = true;
    for(int v = 0; v < kNumNodes; v ++){
        if(graph[u][v] == 0) continue;
        if(dfn[v] == 0) {
            Tarjan(v);
            low[u] = min(low[u], low[v]);
        } else if(in_stack[v] && dfn[v] < low[u]) {
            low[u] = dfn[v];
        }
    }

    if(low[u] == dfn[u]) {
        // pop SCC nodes to a vector 
        vector<int> scc;
        int v;
        do{
            v = st.top();
            st.pop();
            in_stack[v] = false;
            scc.push_back(v);
        }while(u != v);
        scc_list.push_back(scc);
    }
}


void GetSCCs() {
    for(int i = 0; i < kNumNodes; i++) {
        if(!dfn[i]) Tarjan(i);
    }
}

// bool cmp(const vector<int>& a, const vector<int>& b) {
//     return a.size() < b.size();
// }

void SaveSCCsToTXT(string filename){
    ofstream out;
    out.open(filename, ios::out);
    if(out.is_open()) {
        for(auto list : scc_list) {
            for(auto u : list) {
                out << u << " "; 
            }
            out << endl;
        }
        out.close();
        cout << "Finish saving all the SCCs to " + filename << endl;
    } else {
        cerr << "Failed to open file " + filename <<endl;
    }
}

