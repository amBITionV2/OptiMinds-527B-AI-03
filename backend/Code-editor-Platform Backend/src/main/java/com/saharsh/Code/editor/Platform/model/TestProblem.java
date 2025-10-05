package com.saharsh.Code.editor.Platform.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Entity
@NoArgsConstructor
@AllArgsConstructor
public class TestProblem {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int Id;

    @ManyToOne
    @JoinColumn(name = "testId")
    @JsonIgnore
    private Test test;

    @ManyToOne
    @JoinColumn(name="qId")
    private Question question;

    private int points;

    private int orderIdTest;
}
