package com.saharsh.Code.editor.Platform.service;
import com.saharsh.Code.editor.Platform.model.Users;
import com.saharsh.Code.editor.Platform.repo.UsersRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UsersService {

    @Autowired
    private UsersRepository usersRepository;

    public List<Users> getAllUsers() {
        return usersRepository.findAll();
    }

    public Optional<Users> getUserById(int id) {
        return usersRepository.findById(id);
    }

    public Users saveUser(Users user) {
        return usersRepository.save(user);
    }

    public void deleteUser(int id) {
        usersRepository.deleteById(id);
    }

    public Users getUserByUsername(String username) {
        return usersRepository.findByUsername(username);
    }
}
