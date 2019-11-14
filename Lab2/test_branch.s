	.file	"test_branch.c"
	.text
.globl get_vec_element
	.type	get_vec_element, @function
get_vec_element:
.LFB23:
	.cfi_startproc
	testq	%rsi, %rsi
	js	.L2
	cmpq	(%rdi), %rsi
	jge	.L2
	movq	8(%rdi), %rax
	movq	(%rax,%rsi,8), %rax
	movq	%rax, (%rdx)
	movl	$1, %eax
	ret
.L2:
	movl	$0, %eax
	ret
	.cfi_endproc
.LFE23:
	.size	get_vec_element, .-get_vec_element
.globl get_vec_length
	.type	get_vec_length, @function
get_vec_length:
.LFB24:
	.cfi_startproc
	movq	(%rdi), %rax
	ret
	.cfi_endproc
.LFE24:
	.size	get_vec_length, .-get_vec_length
.globl set_vec_length
	.type	set_vec_length, @function
set_vec_length:
.LFB25:
	.cfi_startproc
	movq	%rsi, (%rdi)
	movl	$1, %eax
	ret
	.cfi_endproc
.LFE25:
	.size	set_vec_length, .-set_vec_length
.globl init_vector1
	.type	init_vector1, @function
init_vector1:
.LFB26:
	.cfi_startproc
	movl	$0, %eax
	testq	%rsi, %rsi
	jle	.L11
	movq	%rsi, (%rdi)
	movl	$0, %eax
.L12:
	leaq	0(,%rax,8), %rdx
	addq	8(%rdi), %rdx
	addq	$1, %rax
	cvtsi2sdq	%rax, %xmm0
	movsd	%xmm0, (%rdx)
	cmpq	%rsi, %rax
	jne	.L12
	movl	$1, %eax
.L11:
	rep; ret
	.cfi_endproc
.LFE26:
	.size	init_vector1, .-init_vector1
.globl init_vector2
	.type	init_vector2, @function
init_vector2:
.LFB27:
	.cfi_startproc
	movl	$0, %eax
	testq	%rsi, %rsi
	jle	.L17
	movq	%rsi, (%rdi)
	movl	$0, %eax
.L18:
	movq	8(%rdi), %rdx
	cvtsi2sdq	%rax, %xmm0
	movsd	%xmm0, (%rdx,%rax,8)
	addq	$1, %rax
	cmpq	%rsi, %rax
	jne	.L18
	movl	$1, %eax
.L17:
	rep; ret
	.cfi_endproc
.LFE27:
	.size	init_vector2, .-init_vector2
.globl get_vec_start
	.type	get_vec_start, @function
get_vec_start:
.LFB28:
	.cfi_startproc
	movq	8(%rdi), %rax
	ret
	.cfi_endproc
.LFE28:
	.size	get_vec_start, .-get_vec_start
.globl diff
	.type	diff, @function
diff:
.LFB29:
	.cfi_startproc
	movq	%rdx, %rax
	movq	%rcx, %rdx
	subq	%rsi, %rdx
	jns	.L24
	subq	$1, %rax
	subq	%rdi, %rax
	leaq	1000000000(%rcx), %rdx
	subq	%rsi, %rdx
	ret
.L24:
	subq	%rdi, %rax
	ret
	.cfi_endproc
.LFE29:
	.size	diff, .-diff
.globl branch1
	.type	branch1, @function
branch1:
.LFB31:
	.cfi_startproc
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdi, %rbx
	movq	%rsi, %rbp
	movq	%rdx, %r13
	call	get_vec_length
	movq	%rax, %r12
	movq	%rbx, %rdi
	call	get_vec_start
	movq	%rax, %rbx
	movq	%rbp, %rdi
	call	get_vec_start
	movq	%rax, %rbp
	movq	%r13, %rdi
	call	get_vec_start
	testq	%r12, %r12
	jle	.L33
	movl	$0, %edx
.L32:
	movsd	(%rbx,%rdx,8), %xmm0
	movsd	0(%rbp,%rdx,8), %xmm1
	maxsd	%xmm1, %xmm0
	movsd	%xmm0, (%rax,%rdx,8)
	addq	$1, %rdx
	cmpq	%r12, %rdx
	jne	.L32
.L33:
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE31:
	.size	branch1, .-branch1
.globl branch2
	.type	branch2, @function
branch2:
.LFB32:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	movq	%rsi, %r13
	movq	%rdx, %r14
	call	get_vec_length
	movq	%rax, %r12
	movq	%rbx, %rdi
	call	get_vec_start
	movq	%rax, %rbp
	movq	%r13, %rdi
	call	get_vec_start
	movq	%rax, %rbx
	movq	%r14, %rdi
	call	get_vec_start
	testq	%r12, %r12
	jle	.L40
	movl	$0, %edx
.L39:
	movsd	0(%rbp,%rdx,8), %xmm0
	movsd	(%rbx,%rdx,8), %xmm1
	maxsd	%xmm1, %xmm0
	movsd	%xmm0, (%rax,%rdx,8)
	addq	$1, %rdx
	cmpq	%r12, %rdx
	jne	.L39
.L40:
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE32:
	.size	branch2, .-branch2
.globl fRand
	.type	fRand, @function
fRand:
.LFB30:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movsd	%xmm0, (%rsp)
	movsd	%xmm1, 8(%rsp)
	call	random
	movsd	8(%rsp), %xmm1
	subsd	(%rsp), %xmm1
	cvtsi2sdq	%rax, %xmm0
	divsd	.LC0(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	(%rsp), %xmm0
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE30:
	.size	fRand, .-fRand
.globl new_vec
	.type	new_vec, @function
new_vec:
.LFB22:
	.cfi_startproc
	movq	%rbx, -16(%rsp)
	movq	%rbp, -8(%rsp)
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -16
	.cfi_offset 3, -24
	movq	%rdi, %rbp
	movl	$16, %edi
	call	malloc
	movq	%rax, %rbx
	testq	%rax, %rax
	je	.L45
	movq	%rbp, (%rax)
	testq	%rbp, %rbp
	jle	.L46
	movl	$8, %esi
	movq	%rbp, %rdi
	call	calloc
	testq	%rax, %rax
	jne	.L47
	movq	%rbx, %rdi
	call	free
	movl	$0, %ebx
	jmp	.L45
.L47:
	movq	%rax, 8(%rbx)
	jmp	.L45
.L46:
	movq	$0, 8(%rax)
.L45:
	movq	%rbx, %rax
	movq	8(%rsp), %rbx
	movq	16(%rsp), %rbp
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE22:
	.size	new_vec, .-new_vec
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC1:
	.string	"\n Hello World -- psum examples"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"\n%d,  "
.LC3:
	.string	", "
.LC5:
	.string	"%ld"
	.text
.globl main
	.type	main, @function
main:
.LFB21:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$1384, %rsp
	.cfi_def_cfa_offset 1440
	movl	$.LC1, %edi
	call	puts
	movl	$21000, %edi
	call	new_vec
	movq	%rax, %rbx
	movl	$21000, %edi
	call	new_vec
	movq	%rax, %rbp
	movl	$21000, %edi
	call	new_vec
	movq	%rax, %r12
	movl	$21000, %esi
	movq	%rbx, %rdi
	call	init_vector1
	movl	$21000, %esi
	movq	%rbp, %rdi
	call	init_vector1
	movl	$1000, %r13d
	movl	$0, %r14d
.L50:
	leaq	1(%r14), %r15
	movq	%r13, %rsi
	movq	%rbx, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%rbp, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%r12, %rdi
	call	set_vec_length
	leaq	1360(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%rbx, %rdi
	call	branch1
	leaq	1344(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	salq	$4, %r14
	leaq	(%rsp,%r14), %r14
	movq	1344(%rsp), %rdx
	movq	1352(%rsp), %rcx
	movq	1360(%rsp), %rdi
	movq	1368(%rsp), %rsi
	call	diff
	movq	%rax, (%r14)
	movq	%rdx, 8(%r14)
	addq	$1000, %r13
	movq	%r15, %r14
	cmpq	$20, %r15
	jne	.L50
	movl	$21000, %esi
	movq	%rbx, %rdi
	call	init_vector2
	movl	$21000, %esi
	movq	%rbp, %rdi
	call	init_vector2
	movl	$1000, %r13d
	movb	$0, %r14b
.L51:
	leaq	1(%r14), %r15
	movq	%r13, %rsi
	movq	%rbx, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%rbp, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%r12, %rdi
	call	set_vec_length
	leaq	1360(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%rbx, %rdi
	call	branch1
	leaq	1344(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	salq	$4, %r14
	leaq	336(%rsp,%r14), %r14
	movq	1344(%rsp), %rdx
	movq	1352(%rsp), %rcx
	movq	1360(%rsp), %rdi
	movq	1368(%rsp), %rsi
	call	diff
	movq	%rax, (%r14)
	movq	%rdx, 8(%r14)
	addq	$1000, %r13
	movq	%r15, %r14
	cmpq	$20, %r15
	jne	.L51
	movl	$21000, %esi
	movq	%rbx, %rdi
	call	init_vector1
	movl	$21000, %esi
	movq	%rbp, %rdi
	call	init_vector1
	movl	$1000, %r13d
	movb	$0, %r14b
.L52:
	leaq	1(%r14), %r15
	movq	%r13, %rsi
	movq	%rbx, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%rbp, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%r12, %rdi
	call	set_vec_length
	leaq	1360(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%rbx, %rdi
	call	branch2
	leaq	1344(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	salq	$4, %r14
	leaq	672(%rsp,%r14), %r14
	movq	1344(%rsp), %rdx
	movq	1352(%rsp), %rcx
	movq	1360(%rsp), %rdi
	movq	1368(%rsp), %rsi
	call	diff
	movq	%rax, (%r14)
	movq	%rdx, 8(%r14)
	addq	$1000, %r13
	movq	%r15, %r14
	cmpq	$20, %r15
	jne	.L52
	movl	$21000, %esi
	movq	%rbx, %rdi
	call	init_vector2
	movl	$21000, %esi
	movq	%rbp, %rdi
	call	init_vector2
	movl	$1000, %r13d
	movb	$0, %r14b
.L53:
	leaq	1(%r14), %r15
	movq	%r13, %rsi
	movq	%rbx, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%rbp, %rdi
	call	set_vec_length
	movq	%r13, %rsi
	movq	%r12, %rdi
	call	set_vec_length
	leaq	1360(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%rbx, %rdi
	call	branch2
	leaq	1344(%rsp), %rsi
	movl	$2, %edi
	call	clock_gettime
	salq	$4, %r14
	leaq	1008(%rsp,%r14), %r14
	movq	1344(%rsp), %rdx
	movq	1352(%rsp), %rcx
	movq	1360(%rsp), %rdi
	movq	1368(%rsp), %rsi
	call	diff
	movq	%rax, (%r14)
	movq	%rdx, 8(%r14)
	addq	$1000, %r13
	movq	%r15, %r14
	cmpq	$20, %r15
	jne	.L53
	jmp	.L68
.L58:
	movq	%r12, %rbx
.L60:
	leaq	1(%rbx), %r12
	movl	%r13d, %esi
	movl	$.LC2, %edi
	movl	$0, %eax
	call	printf
	salq	$4, %rbx
	leaq	(%r14,%rbx), %rbx
	movq	%r15, %rbp
	jmp	.L55
.L57:
	addq	$1, %rbp
	addq	$336, %rbx
.L55:
	cmpq	$1, %rbp
	je	.L56
	movl	$.LC3, %edi
	movl	$0, %eax
	call	printf
	imulq	$1000000000, (%rbx), %rax
	addq	8(%rbx), %rax
	cvtsi2sdq	%rax, %xmm0
	mulsd	.LC4(%rip), %xmm0
	cvttsd2siq	%xmm0, %rsi
	movl	$.LC5, %edi
	movl	$0, %eax
	call	printf
	cmpq	$3, %rbp
	jle	.L57
	addl	$1000, %r13d
	cmpq	$20, %r12
	jne	.L58
	jmp	.L69
.L68:
	movl	$1000, %r13d
	movl	$0, %ebx
	movq	%rsp, %r14
	movl	$1, %r15d
	jmp	.L60
.L69:
	movl	$10, %edi
	call	putchar
	movl	$0, %eax
	addq	$1384, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L56:
	.cfi_restore_state
	imulq	$1000000000, (%rbx), %rax
	addq	8(%rbx), %rax
	cvtsi2sdq	%rax, %xmm0
	mulsd	.LC4(%rip), %xmm0
	cvttsd2siq	%xmm0, %rsi
	movl	$.LC5, %edi
	movl	$0, %eax
	call	printf
	jmp	.L57
	.cfi_endproc
.LFE21:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	4290772992
	.long	1105199103
	.align 8
.LC4:
	.long	858993459
	.long	1074213683
	.ident	"GCC: (GNU) 4.4.7 20120313 (Red Hat 4.4.7-18)"
	.section	.note.GNU-stack,"",@progbits
