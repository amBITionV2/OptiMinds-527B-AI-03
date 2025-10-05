package com.saharsh.Code.editor.Platform.Dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class StudentTestsDTO {
    private int id;
    private int UserId;
    private int score;
    private int TestId;
}
