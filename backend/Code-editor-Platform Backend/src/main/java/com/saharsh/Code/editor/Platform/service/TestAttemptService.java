package com.saharsh.Code.editor.Platform.service;
import com.saharsh.Code.editor.Platform.model.TestAttempt;
import com.saharsh.Code.editor.Platform.repo.TestAttemptRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class TestAttemptService {

    @Autowired
    private TestAttemptRepository testAttemptRepository;

    public List<TestAttempt> getAllTestAttempts() {
        return testAttemptRepository.findAll();
    }

    public Optional<TestAttempt> getTestAttemptById(int id) {
        return testAttemptRepository.findById(id);
    }

    public TestAttempt saveTestAttempt(TestAttempt testAttempt) {
        return testAttemptRepository.save(testAttempt);
    }

    public void deleteTestAttempt(int id) {
        testAttemptRepository.deleteById(id);
    }
}