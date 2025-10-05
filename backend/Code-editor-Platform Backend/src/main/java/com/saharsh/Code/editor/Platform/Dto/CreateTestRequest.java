package com.saharsh.Code.editor.Platform.Dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.sql.Timestamp;
import java.util.List;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class CreateTestRequest {
    private String testName;
    private String description;
    private Integer durationMinutes;
    private Timestamp startTime;
    private Timestamp endTime;
    private List<TestProblemDTO> questions;
    private int companyId;
}
