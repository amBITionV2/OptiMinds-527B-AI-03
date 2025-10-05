package com.saharsh.Code.editor.Platform.service;

import com.saharsh.Code.editor.Platform.model.TestCase;
import com.saharsh.Code.editor.Platform.repo.TestCaseRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class TestCaseService {

    @Autowired
    private TestCaseRepository testCaseRepository;

    public List<TestCase> getAllTestCases() {
        return testCaseRepository.findAll();
    }

    public Optional<TestCase> getTestCaseById(int id) {
        return testCaseRepository.findById(id);
    }

    public TestCase saveTestCase(TestCase testCase) {
        return testCaseRepository.save(testCase);
    }

    public void deleteTestCase(int id) {
        testCaseRepository.deleteById(id);
    }


    public Optional<TestCase> getFirstTestCaseByQuestionId(int questionId) {
        return testCaseRepository.findByQuestionProblemId(questionId);
    }
}
