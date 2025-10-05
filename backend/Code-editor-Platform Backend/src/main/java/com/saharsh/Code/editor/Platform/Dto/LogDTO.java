package com.saharsh.Code.editor.Platform.Dto;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

// DTO for each log entry
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LogDTO {
    private String type;
    private String phase;
    private String ts; // ISO string
    private Long duration;
    private Object details; // can be Map or Object
}


