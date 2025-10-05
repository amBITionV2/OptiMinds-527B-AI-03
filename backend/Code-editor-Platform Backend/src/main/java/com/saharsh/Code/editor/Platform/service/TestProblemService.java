package com.saharsh.Code.editor.Platform.service;

import com.saharsh.Code.editor.Platform.model.TestProblem;
import com.saharsh.Code.editor.Platform.repo.TestProblemRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class TestProblemService {

    @Autowired
    private TestProblemRepository testProblemRepository;

    public List<TestProblem> getAllTestProblems() {
        return testProblemRepository.findAll();
    }

    public Optional<TestProblem> getTestProblemById(int id) {
        return testProblemRepository.findById(id);
    }

    public TestProblem saveTestProblem(TestProblem testProblem) {
        return testProblemRepository.save(testProblem);
    }

    public void deleteTestProblem(int id) {
        testProblemRepository.deleteById(id);
    }
}