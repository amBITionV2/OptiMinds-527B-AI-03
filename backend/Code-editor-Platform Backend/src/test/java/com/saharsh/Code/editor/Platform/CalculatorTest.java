package com.saharsh.Code.editor.Platform;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CalculatorTest {
    //Junit testing;
    Calculator obj = new Calculator();

    int a =0;
    int b = 0;

    @BeforeEach
    void initialize(){
        a=4;
        b = 0;
    }


    @Test
    void condition(){
        assertThrows(IllegalArgumentException.class,() ->{
            a=10;
            b=19;
            obj.divide(a,b);
        });
    }

    @Test
    void checkSum(){
        int result = obj.Add(a,b);

        assertEquals(4,result);
    }
}
