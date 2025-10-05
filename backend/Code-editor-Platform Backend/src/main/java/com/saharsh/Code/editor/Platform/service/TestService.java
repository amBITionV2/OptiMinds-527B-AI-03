package com.saharsh.Code.editor.Platform.service;

import com.saharsh.Code.editor.Platform.Dto.CreateTestRequest;
import com.saharsh.Code.editor.Platform.Dto.TestProblemDTO;
import com.saharsh.Code.editor.Platform.model.*;
import com.saharsh.Code.editor.Platform.repo.ImagesRepository;
import com.saharsh.Code.editor.Platform.repo.TestRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class TestService {

    @Autowired
    private TestRepository testRepository;
    @Autowired
    private ImagesRepository imagesRepository;
    public List<Test> getAllTests() {
        return testRepository.findAll();
    }

    public Optional<Test> getTestById(int id) {
        return testRepository.findById(id);
    }

    public Test saveTest(CreateTestRequest request) {
        Test test = new Test();
        test.setTestName(request.getTestName());
        test.setDescription(request.getDescription());
        test.setDurationMinutes(request.getDurationMinutes());
        test.setStartTime(request.getStartTime());
        test.setEndTime(request.getEndTime());
        Company c = new Company();
        c.setId(1);
        test.setCompany(c);
        List<TestProblem> problems = new ArrayList<>();
        int  i = 0;
        for (TestProblemDTO dto : request.getQuestions()) {
            Question q = new Question();
            TestProblem problem = new TestProblem();
            q.setProblemId(dto.getQuestionId());
            problem.setQuestion(q);
            problem.setPoints(dto.getPoints());
            problem.setTest(test); // link problem to test
            problem.setOrderIdTest(++i);
            problems.add(problem);

        }
        test.setTestProblems(problems);
        return testRepository.save(test);
    }

    public void deleteTest(int id) {
        testRepository.deleteById(id);
    }

    public void saveImage(Long userId, Long testId, MultipartFile frame,Long time,boolean phone) throws Exception {
        if (frame.isEmpty()) throw new Exception("No image provided");
        byte[] imageBytes = frame.getBytes();

        Images image = new Images();
        image.setUserId(userId);
        image.setTestId(testId);
        image.setImageData(imageBytes); // <- must be byte[]
        image.setTime(time);
        image.setIsPhone(phone);

        imagesRepository.save(image);
    }
}