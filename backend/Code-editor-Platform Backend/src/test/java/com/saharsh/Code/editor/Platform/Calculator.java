package com.saharsh.Code.editor.Platform;

public class Calculator {
    public int Add(int a,int b){
        return a+b;
    }

    public int divide(int a,int b) throws Exception{
        if(b!=0)
            return a / b;
        else{
            throw new IllegalArgumentException();
        }
    }
}
