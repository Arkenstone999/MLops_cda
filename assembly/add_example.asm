; Filename: add_example.asm
; Description: Add two constants and exit with the sum as the return code.

section .text
    global _start       ; export entry point

_start:
    ; load immediate values
    mov     rax, 5      ; first addend → rax
    mov     rbx, 3      ; second addend → rbx
    mov     rcx, 5      ; third addend → rcx

    ; perform addition
    add     rax, rbx    ; rax = rax + rbx + rcx (5 + 3 + 5= 12)
    add     rax, rcx    ;
    
    ; exit syscall:
    ;   rax = 60 (sys_exit)
    ;   rdi = return code
    mov     rdi, rax    ; move sum into rdi
    mov     rax, 60     ; syscall number for exit
    syscall             ; invoke kernel
