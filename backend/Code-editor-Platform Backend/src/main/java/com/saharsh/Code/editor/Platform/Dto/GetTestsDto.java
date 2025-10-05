package com.saharsh.Code.editor.Platform.Dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.sql.Timestamp;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class GetTestsDto {
    private int testId;
    private String testName;
    private Timestamp startTime;
    private Timestamp endTime;
    private Timestamp createdAt;
}
