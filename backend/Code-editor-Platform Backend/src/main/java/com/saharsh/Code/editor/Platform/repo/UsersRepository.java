package com.saharsh.Code.editor.Platform.repo;


import com.saharsh.Code.editor.Platform.model.Users;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UsersRepository extends JpaRepository<Users, Integer> {
    Users findByUsername(String username);

    // Optional: method with PESSIMISTIC write lock if you want DB-level locking (uncomment use in service)
    @Query("select u from Users u where u.id = :id")
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    Optional<Users> findByIdForUpdate(@Param("id") Integer id);

    Users findById(Long userId);
}
