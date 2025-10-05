package com.saharsh.Code.editor.Platform.Dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserActivityDTO {
    private Integer userId;
    private String testId;
    private List<LogDTO> logs;
}
