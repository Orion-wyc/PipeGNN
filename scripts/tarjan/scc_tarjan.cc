/*  Date: 2020/11/26
 *  Author: Yuchen 
 *  Descripion: tarjan.cc is used to get all the SCCs(Strongly Connected Components) of a given Graph G.
 *              In tarjan.cc, Graph is represented by Adjacency Matrix. 
 *              [yes] It will be changed to Adjacency List in next version.
 *              [] Use class to do this!!
 * 
 *  Tips: You are probably redefining the default parameter in the definition of the function AddEdge(). 
          It should only be defined in the function declaration below.
          https://blog.csdn.net/wangshubo1989/article/details/50135039
 */


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stack>
#include <algorithm>
#include <unordered_set>

using namespace std;


//QAQ these will be deprecated in next version 
const int kNumNodes =  9e5; //7115-wikivote

// data path: maybe need to change it in Windows QAQ
string basepath = "./data/";
const string infilename = "wiki-vote7115.txt";
const string outfilename = "wiki-result.txt";


// used in function TarjanSCC
struct Edge {
    int from, to, value;
    Edge(int u, int v, int val) : from(u), to(v), value(val){}
};
vector<Edge> edges;
vector<int> graph[kNumNodes]; //use Adjacency List is better

int max_node_num = 0;

int dfn_index = 0; // (timestamp)
int low[kNumNodes]; // The root number of the smallest subtree which contains Node u.
int dfn[kNumNodes]; // The order in which nodes are searched during depth-first search traversal.
bool in_stack[kNumNodes];
stack<int> st;
vector< vector<int> > scc_list; // save all the SCCs


// functions' declaration: default parameters declare once
void InitArrays();
void AddEdge(int u, int v, int val=0);
void BuildGraphFromTXT(string filepath);
void Tarjan(int u);
void GetSCCs();
bool Compare(const vector<int>& a, const vector<int>& b);
void SaveSCCsToTXT(string filename);
void AnalyzeSCC();


int main() {
    InitArrays();
    // use python to preprocess "wiki-vote7115.txt"
    BuildGraphFromTXT(basepath + infilename); 
    GetSCCs();
    // analyzing before saving
    AnalyzeSCC(); 
    // "wiki-SCCs.txt"
    SaveSCCsToTXT(basepath + outfilename); 
    return 0;
}


void InitArrays() {
    cout << "Init..." <<endl;
    for(int i = 0; i < kNumNodes; i ++) {
        dfn[i] = 0;
        in_stack[i] = false;
        graph[i].clear();
    }
    edges.clear();
}


void AddEdge(int u, int v, int val /*=0*/) {
    edges.push_back(Edge(u, v, val));
    graph[u].push_back(edges.size()-1);
}


void BuildGraphFromTXT(string filename) {
    ifstream in;
    in.open(filename, ios::in);
    if(in.is_open()){ //be cautious here: filename inexistance will cause Endless loop
        int u, v;
        int cnt = 0;
        // int max_node_num = 0;
        while(!in.eof()) {
            in >> u >> v;
            AddEdge(u, v);
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
    for(int i = 0; i < graph[u].size(); i ++){
        Edge e = edges[graph[u][i]];
        int v = e.to;
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


bool Compare(const vector<int>& a, const vector<int>& b) {
    return a.size() > b.size();
}


/* Print current graph's largest SCC's infomation */
void AnalyzeSCC() {
    // sorted by number of nodes
    sort(scc_list.begin(), scc_list.end(), Compare);
    
    int max_nodes  = 0;
    int max_edges = 0;
    vector<int> nodes;

    nodes = scc_list[0];
    unordered_set<int> scc_set(nodes.begin(), nodes.end());
    max_nodes = nodes.size();
    // count edges in the largest SCC
    for (auto &u : nodes) {
        for (auto &edge_id : graph[u]){
            if(scc_set.count(edges[edge_id].to) > 0) {               
                max_edges += 1;
            } 
        }     
    }
    // for (auto &e : edges) {
    //     if (scc_set.count(e.from) > 0 && scc_set.count(e.to) >0) {
    //         max_edges += 1;
    //     }
    // }
    //print analyzing results
    cout << "Nodes in largest SCC (ratio): " << max_nodes << "(" << max_nodes / (1.0*max_node_num)  << ")" << endl;
    cout << "Edges in largest SCC (ratio): " << max_edges << "(" << max_edges / (1.0*edges.size()) << ")" << endl;

    // add some more features here, e.g. count SCCs
}


/* remove sort in writing scc, the scc_list is sorted */
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


