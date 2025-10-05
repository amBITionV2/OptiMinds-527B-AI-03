package com.saharsh.Code.editor.Platform.Dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TestProblemDTO {
    @JsonProperty("problem_id")   // maps snake_case JSON to camelCase field
    private Integer questionId;
    private Integer points;
    private String difficulty;
    private int orderIdTest;
}
