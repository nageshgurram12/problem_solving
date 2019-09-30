class Solution {
    public String longestPalindrome(String s) {
        int len = s.length();
        if(len == 0){
            return "";
        }
        boolean[][] A = new boolean[len][len];
        
        // Initialize A with all one char strings
        int maxLen = 1;
        String maxPalString = s.substring(0,1);
        
        for(int i=0; i<len; i++){
            A[i][i] = true;
            if(i<len-1 && s.charAt(i) == s.charAt(i+1)){
                A[i][i+1] = true;
                maxLen = 2;
                maxPalString = s.substring(i,i+2);
            }
        }
        
        // Iterate over lengths of strings
        for(int l=3; l<=len; l++){
            for(int j=0; j<=len-l; j++){
                int k = j + (l-1);
                A[j][k] = A[j+1][k-1] && s.charAt(j) == s.charAt(k);
                if(A[j][k] && l >= maxLen){
                    maxLen = l;
                    maxPalString = s.substring(j, j+l);
                }   
            }
        }
        return maxPalString;
    }
}
