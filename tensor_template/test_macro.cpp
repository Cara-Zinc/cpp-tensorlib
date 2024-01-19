#include <iostream>
#include <unordered_map>
using namespace std;
std::string standardize_einsum_notation(const std::string& equation);
int main()
{
    std::string str = "j,b->jb";
    cout<<standardize_einsum_notation(str);
    return 0;
}

std::string standardize_einsum_notation(const std::string& equation) {
        std::unordered_map<char, char> index_map;
        char current_replacement = 'i';
        
        // Replace each unique index with 'i', 'j', 'k', ...
        for (char c : equation) {
            if (isalpha(c) && index_map.find(c) == index_map.end()) {
                index_map[c] = current_replacement++;
            }
        }

        std::string standardized_equation;
        for (char c : equation) {
            if (isalpha(c)) {
                standardized_equation += index_map[c];
            } else {
                standardized_equation += c;
            }
        }

        return standardized_equation;
    }